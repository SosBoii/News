"""
Personal news feed builder.

Reads config.yaml, pulls RSS feeds / Google News searches, filters by keywords,
merges duplicate stories across sites, picks images and writes feed.json for index.html.

Usage:
    python fetch.py                      # config.yaml -> feed.json
    python fetch.py --config other.yaml --out other.json   # second profile
    python fetch.py --offline sample     # no network: build from tests/sample_feed.xml
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, quote_plus, urlparse

import feedparser
import requests
import yaml

log = logging.getLogger("news-feed")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 personal-news-feed"
)
HTTP_TIMEOUT = 15
MAX_ARTICLES_PER_TAB_STORED = 60  # cap on what we keep per tab across runs (feed.json size)
MAX_ALSO = 5                      # how many "also reported by" publishers to keep per card
MAX_AI_RETRY = 80                 # stored articles without AI review re-sent per run (Gemini was down before)

Article = Dict[str, Any]


# ============================================================================
# Config
# ============================================================================

DEFAULT_SETTINGS = {
    "max_age_hours": 72,
    "per_source": 3,
    "important_min_sources": 3,
    "similarity": 0.45,
    "fetch_images": True,
    "max_image_fetches": 60,
    "google_window": "when:2d",
}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["settings"] = {**DEFAULT_SETTINGS, **(cfg.get("settings") or {})}
    kw = cfg.get("keywords") or {}
    cfg["keywords"] = {k: [str(w) for w in (kw.get(k) or [])] for k in ("block", "allow", "quiet")}
    cfg["tabs"] = cfg.get("tabs") or []
    seen = set()
    for t in cfg["tabs"]:
        if not t.get("id") or t["id"] in seen:
            raise ValueError("every tab needs a unique id (problem with: %r)" % t.get("name"))
        seen.add(t["id"])
        t.setdefault("name", t["id"])
        t.setdefault("emoji", "📰")
        t.setdefault("max", 8)
        t["sources"] = t.get("sources") or []
    return cfg


# ============================================================================
# Text helpers
# ============================================================================

_QUOTES = str.maketrans({"״": '"', "”": '"', "“": '"', "„": '"', "׳": "'", "’": "'", "‘": "'", "`": "'"})
_NIQQUD = re.compile(r"[\u0591-\u05C7]")
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, unify Hebrew/English quote marks, drop niqqud, collapse whitespace."""
    text = html.unescape(text or "").translate(_QUOTES)
    text = _NIQQUD.sub("", text)
    return _WS.sub(" ", text).strip().lower()


def strip_html(text: str) -> str:
    return _WS.sub(" ", html.unescape(_TAGS.sub(" ", text or ""))).strip()


# Hebrew one-letter prefixes (ו, ה, ב, ל, מ, ש, כ) may be glued to a word: "והליכוד", "בכנסת".
_HE_PREFIX = "[ובהלמשכ]{0,3}"


def _keyword_regex(word: str) -> re.Pattern:
    w = re.escape(normalize(word))
    w = w.replace(r"\ ", r"\s+")
    return re.compile(r"(?<!\w)" + _HE_PREFIX + w + r"(?!\w)")


class KeywordFilter:
    """block / allow / quiet keyword lists. Matching is on title + summary."""

    def __init__(self, keywords: Dict[str, List[str]]):
        self.block = [(_keyword_regex(w), w) for w in keywords.get("block", []) if str(w).strip()]
        self.allow = [(_keyword_regex(w), w) for w in keywords.get("allow", []) if str(w).strip()]
        self.quiet = [(_keyword_regex(w), w) for w in keywords.get("quiet", []) if str(w).strip()]

    @staticmethod
    def _first(patterns, text: str) -> Optional[str]:
        for rx, word in patterns:
            if rx.search(text):
                return word
        return None

    def classify(self, title: str, summary: str = "") -> Tuple[str, Optional[str], bool]:
        """
        Returns (verdict, matched_word, quiet):
            verdict: "allow"  - an always-pass word matched (wins over block)
                     "block"  - a blocked word matched
                     "ok"     - nothing special
        """
        text = normalize(title + " \n " + summary)
        quiet = self._first(self.quiet, text) is not None
        allowed = self._first(self.allow, text)
        if allowed:
            return "allow", allowed, quiet
        blocked = self._first(self.block, text)
        if blocked:
            return "block", blocked, quiet
        return "ok", None, quiet


# ============================================================================
# Fetching
# ============================================================================

def bing_news_url(query: str, lang: str = "en") -> str:
    mkt = "he-IL" if lang == "he" else "en-US"
    return "https://www.bing.com/news/search?q=%s&format=rss&mkt=%s" % (quote_plus(query), mkt)


def unwrap_bing_link(link: str) -> str:
    """Bing RSS links go through bing.com/news/apiclick.aspx?...&url=<real url>. Return the real article URL."""
    try:
        p = urlparse(link)
        if p.netloc.endswith("bing.com") and "apiclick" in p.path:
            real = parse_qs(p.query).get("url", [""])[0]
            if real.startswith("http"):
                return real
    except ValueError:
        pass
    return link


def google_news_url(query: str, lang: str = "he", window: str = "") -> str:
    q = (query + " " + window).strip() if window else query
    if lang == "en":
        loc = "hl=en-US&gl=US&ceid=US:en"
    else:
        loc = "hl=he&gl=IL&ceid=IL:he"
    return "https://news.google.com/rss/search?q=%s&%s" % (quote_plus(q), loc)


def http_get(url: str, timeout: int = HTTP_TIMEOUT) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "he,en;q=0.8"}, timeout=timeout)
    r.raise_for_status()
    return r


def parse_feed_bytes(data: bytes) -> List[Any]:
    parsed = feedparser.parse(data)
    return list(parsed.entries or [])


_IMG_IN_HTML = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.I)


def entry_image(entry: Any) -> str:
    bing_img = entry.get("news_image")  # Bing: <News:Image>
    if bing_img and str(bing_img).startswith("http"):
        url = str(bing_img).replace("http://", "https://", 1)
        if "bing.com/th" in url:
            # ask Bing's thumbnail service for a card-sized image instead of the tiny default
            url += ("&" if "?" in url else "?") + "w=700&h=394&c=7"
        return url
    for key in ("media_content", "media_thumbnail"):
        for m in entry.get(key) or []:
            url = m.get("url")
            if url and (m.get("medium") in (None, "image") or "image" in str(m.get("type", ""))):
                return url
    for enc in entry.get("enclosures") or []:
        if "image" in str(enc.get("type", "")) and enc.get("href"):
            return enc["href"]
    for field in ("summary", "description"):
        m = _IMG_IN_HTML.search(entry.get(field) or "")
        if m and not m.group(1).startswith("data:"):
            return m.group(1)
    for c in entry.get("content") or []:
        m = _IMG_IN_HTML.search(c.get("value") or "")
        if m:
            return m.group(1)
    return ""


def entry_time(entry: Any, now: datetime) -> datetime:
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            try:
                dt = datetime(*st[:6], tzinfo=timezone.utc)
                return min(dt, now)  # some feeds publish future timestamps
            except (TypeError, ValueError):
                pass
    return now


def article_id(link: str, title: str) -> str:
    return hashlib.sha1((link or title).encode("utf-8")).hexdigest()[:12]


def entry_to_article(entry: Any, tab_id: str, source: Dict[str, Any], via: str, now: datetime) -> Optional[Article]:
    title = strip_html(entry.get("title") or "")
    link = (entry.get("link") or "").strip()
    if not title or not link:
        return None
    source_name = source.get("name") or urlparse(link).netloc
    summary_html = entry.get("summary") or entry.get("description") or ""
    summary = strip_html(summary_html)

    if via == "bing":
        link = unwrap_bing_link(link)
        publisher = str(entry.get("news_source") or "").strip()
        source_name = publisher or urlparse(link).netloc.replace("www.", "") or source_name

    if via == "google":
        # Google News titles look like: "Headline - Publisher"
        src = entry.get("source") or {}
        publisher = (src.get("title") if hasattr(src, "get") else "") or ""
        if not publisher and " - " in title:
            publisher = title.rsplit(" - ", 1)[1]
        if publisher and title.endswith(" - " + publisher):
            title = title[: -len(" - " + publisher)].strip()
        if publisher:
            source_name = publisher.strip()
        # Google's description just repeats the headline + publisher links – useless
        summary = ""

    if summary and normalize(summary).startswith(normalize(title)[:40]):
        summary = summary[len(title):].strip(" -–:")

    return {
        "id": article_id(link, title),
        "tab": tab_id,
        "feed": source.get("_key", source_name),   # which configured source produced it
        "title": title,
        "summary": summary[:400],
        "link": link,
        "source": source_name,
        "lang": source.get("lang", "he"),
        "image": entry_image(entry),
        "published": entry_time(entry, now).isoformat(),
    }


def fetch_source(
    source: Dict[str, Any],
    tab_id: str,
    settings: Dict[str, Any],
    now: datetime,
    getter: Callable[[str], bytes],
) -> Tuple[List[Article], Dict[str, Any]]:
    """
    Try in order: the site's RSS url -> Bing News search -> Google News search.
    The first one that returns articles wins. Bing gives images and direct links; Google is the safety net.
    """
    attempts: List[Tuple[str, str]] = []
    lang = source.get("lang", "he")
    if source.get("url"):
        attempts.append(("rss", source["url"]))
    if source.get("bing"):
        attempts.append(("bing", bing_news_url(source["bing"], lang)))
    if source.get("google"):
        attempts.append(("google", google_news_url(source["google"], lang, settings.get("google_window", ""))))

    status = {"tab": tab_id, "name": source.get("name", "?"), "ok": False, "via": None, "count": 0, "error": None}
    errors = []
    for via, url in attempts:
        try:
            entries = parse_feed_bytes(getter(url))
        except Exception as e:  # network / http / parse – never fail the whole run
            errors.append("%s: %s" % (via, str(e)[:120]))
            continue
        arts = [a for a in (entry_to_article(e, tab_id, source, via, now) for e in entries) if a]
        if arts:
            status.update(ok=True, via=via, count=len(arts))
            return arts, status
        errors.append("%s: empty" % via)
    status["error"] = "; ".join(errors) or "no url/google configured"
    return [], status


# ============================================================================
# Filtering, clustering, ranking
# ============================================================================

_TOKEN = re.compile(r"\w+", re.U)
_STOP = {
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is", "are", "at", "by", "with", "from",
    "של", "את", "על", "עם", "גם", "לא", "כי", "זה", "זו", "הוא", "היא", "הם", "אחרי", "לפני", "בין", "כל",
}


def title_tokens(title: str) -> set:
    toks = set()
    for t in _TOKEN.findall(normalize(title)):
        if t in _STOP or len(t) < 2:
            continue
        # crude Hebrew prefix stripping so "בירושלים" ~ "ירושלים"
        if len(t) > 3 and t[0] in "ובהלמשכ" and re.match(r"[\u0590-\u05FF]", t):
            t = t[1:]
        toks.add(t)
    return toks


def similarity(a: set, b: set) -> float:
    """Dice coefficient of title tokens: 1.0 = same words, 0.0 = nothing in common."""
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def cluster(articles: List[Article], threshold: float) -> List[Article]:
    """
    Merge articles that describe the same story (similar titles).
    The earliest-published article becomes the card; others are listed in `also`.
    """
    ordered = sorted(articles, key=lambda a: a["published"])
    clusters: List[Tuple[set, Article]] = []
    for art in ordered:
        toks = title_tokens(art["title"])
        for ctoks, head in clusters:
            if art["link"] == head["link"] or similarity(toks, ctoks) >= threshold:
                if art["source"] != head["source"] and art["source"] not in head["also"] \
                        and len(head["also"]) < MAX_ALSO:
                    head["also"].append(art["source"])
                feed = art.get("feed")
                if feed and feed not in head["feeds"]:
                    head["feeds"].append(feed)
                if not head.get("image") and art.get("image"):
                    head["image"] = art["image"]
                if not head.get("summary") and art.get("summary"):
                    head["summary"] = art["summary"]
                break
        else:
            head = dict(art, also=list(art.get("also") or []),
                        feeds=list(art.get("feeds") or ([art["feed"]] if art.get("feed") else [])))
            clusters.append((toks, head))
    return [h for _, h in clusters]


def rank_and_cap(articles: List[Article], tab: Dict[str, Any], settings: Dict[str, Any],
                 factor: float = 1.0) -> List[Article]:
    """
    Sort (multi-source first, quiet last, newest first) and apply per-source and per-tab caps.
    factor > 1 keeps a bigger pool (e.g. 2x) so the AI has room to remove items and the tab still fills up.
    """
    per_source = int(tab.get("per_source") or settings.get("per_source") or 3)
    limit = int(tab.get("max") or 8)
    if factor > 1:
        limit = int(round(limit * factor))
        per_source = int(round(per_source * factor))

    def key(a: Article):
        # quiet last; stories several of *our* sources reported first; then newest
        return (bool(a.get("quiet")), -min(int(a.get("multi") or 1), 3), _neg_time(a["published"]))

    out, counts = [], {}
    for a in sorted(articles, key=key):
        src = a["source"]
        if counts.get(src, 0) >= per_source:
            continue
        counts[src] = counts.get(src, 0) + 1
        out.append(a)
        if len(out) >= limit:
            break
    return out


def final_cap(articles: List[Article], tabs: List[Dict[str, Any]], settings: Dict[str, Any],
              already: Optional[Dict[str, int]] = None) -> List[Article]:
    """
    After the AI: cut each tab back to its `max` and each source back to `per_source`.
    Prefer 🔴, then non-quiet, then higher AI score; ties keep the original ranking order.
    Articles without a score (AI unavailable) count as 3. "Top 5" is renumbered among survivors.
    """
    by_tab = {t["id"]: t for t in tabs}
    order = {id(a): i for i, a in enumerate(articles)}
    ranked = sorted(articles, key=lambda a: (not a.get("important"), bool(a.get("quiet")),
                                             -int(a.get("score") or 3), order[id(a)]))
    tab_counts: Dict[str, int] = {}
    src_counts: Dict[Tuple[str, str], int] = {}
    keep = set()
    for a in ranked:
        tab = by_tab.get(a["tab"], {})
        # stories from this scan that are already in the feed use up part of the tab's quota
        limit = max(0, int(tab.get("max") or 8) - (already or {}).get(a["tab"], 0))
        per_source = int(tab.get("per_source") or settings.get("per_source") or 3)
        sk = (a["tab"], a["source"])
        if tab_counts.get(a["tab"], 0) >= limit or src_counts.get(sk, 0) >= per_source:
            continue
        tab_counts[a["tab"]] = tab_counts.get(a["tab"], 0) + 1
        src_counts[sk] = src_counts.get(sk, 0) + 1
        keep.add(id(a))
    out = [a for a in articles if id(a) in keep]
    if any("score" in a for a in out):  # AI ran – recompute "5 things" among the survivors only
        for a in out:
            a.pop("top", None)
        best = sorted((a for a in out if "score" in a and not a.get("quiet")),
                      key=lambda a: (not a.get("important"), -a["score"], order[id(a)]))
        for rank, a in enumerate(best[:5]):
            a["top"] = rank + 1
    return out


def _neg_time(iso: str) -> float:
    try:
        return -datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return 0.0


def mark_important(articles: Iterable[Article], min_sources: int) -> None:
    """
    No longer sets 🔴 – only the AI decides what is important.
    Stories reported by several configured sources just get a `multi` count, used to rank them higher.
    (One Google/Bing search returns many publishers for the same story, so publishers are not counted.)
    """
    for a in articles:
        a["important"] = False
        a["multi"] = len(a.get("feeds") or [])


# ============================================================================
# AI stage (implemented in task 2 – here a pass-through)
# ============================================================================

def ai_stage(kept: List[Article], suspects: List[Article], cfg: Dict[str, Any]) -> Tuple[List[Article], Dict[str, Any]]:
    """
    kept     = articles that passed keyword filtering
    suspects = articles that matched a blocked word (AI may rescue them if very important)
    Without AI, suspects are dropped and kept articles pass unchanged.
    """
    try:
        import ai  # optional module, added in task 2
    except ImportError:
        return kept, {"used": False, "reason": "ai module not installed"}
    return ai.run(kept, suspects, cfg)


# ============================================================================
# Images
# ============================================================================

_OG_IMAGE = re.compile(
    r"<meta[^>]+(?:property|name)=[\"'](?:og:image|twitter:image)[\"'][^>]+content=[\"']([^\"']+)[\"']"
    r"|<meta[^>]+content=[\"']([^\"']+)[\"'][^>]+(?:property|name)=[\"'](?:og:image|twitter:image)[\"']",
    re.I,
)


def find_og_image(page_html: str) -> str:
    m = _OG_IMAGE.search(page_html or "")
    if not m:
        return ""
    return html.unescape(m.group(1) or m.group(2) or "")


def fill_images(articles: List[Article], settings: Dict[str, Any], getter: Callable[[str], bytes]) -> int:
    if not settings.get("fetch_images"):
        return 0
    budget = int(settings.get("max_image_fetches", 0))
    filled = 0
    for a in articles:
        if budget <= 0:
            break
        if a.get("image") or a.get("image_tried"):
            continue
        # Google News links resolve via JavaScript, so the page would only give Google's logo
        if urlparse(a["link"]).netloc.endswith("news.google.com"):
            continue
        budget -= 1
        a["image_tried"] = True
        try:
            img = find_og_image(getter(a["link"])[:300_000].decode("utf-8", "ignore"))
        except Exception as e:
            log.debug("og:image failed for %s: %s", a["link"], e)
            continue
        if img.startswith("//"):
            img = "https:" + img
        if img.startswith("http"):
            a["image"] = img
            filled += 1
    return filled


# ============================================================================
# Merge with previous run + main pipeline
# ============================================================================

def load_previous(path: str) -> List[Article]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return list(json.load(f).get("articles") or [])
    except (OSError, ValueError):
        log.warning("could not read previous %s – starting fresh", path)
        return []


def merge_previous(new: List[Article], previous: List[Article], now: datetime, max_age_hours: int,
                   tab_ids: Iterable[str]) -> List[Article]:
    """
    Keep older articles (so the feed is not empty right after a run), drop expired ones,
    and do not repeat a story that is already in the feed.
    """
    cutoff = now - timedelta(hours=max_age_hours)
    tab_ids = set(tab_ids)
    by_id = {}
    prev_tokens: List[Tuple[set, Article]] = []
    for a in previous:
        if a.get("tab") not in tab_ids:
            continue
        try:
            if datetime.fromisoformat(a.get("first_seen") or a["published"]) < cutoff:
                continue
        except (KeyError, ValueError):
            continue
        by_id[a["id"]] = a
        prev_tokens.append((title_tokens(a["title"]), a))

    for a in new:
        if a["id"] in by_id:
            old = by_id[a["id"]]
            for s in a.get("also") or []:
                if s not in old.setdefault("also", []) and s != old["source"]:
                    old["also"].append(s)
            continue
        toks = title_tokens(a["title"])
        dup = next((p for t, p in prev_tokens if p["tab"] == a["tab"] and similarity(toks, t) >= 0.6), None)
        if dup is not None:
            for s in [a["source"]] + list(a.get("also") or []):
                if s != dup["source"] and s not in dup.setdefault("also", []):
                    dup["also"].append(s)
            continue
        a["first_seen"] = now.isoformat()
        by_id[a["id"]] = a
        prev_tokens.append((toks, a))

    merged = list(by_id.values())
    merged.sort(key=lambda a: a.get("first_seen") or a["published"], reverse=True)
    # cap per tab
    out, counts = [], {}
    for a in merged:
        counts[a["tab"]] = counts.get(a["tab"], 0) + 1
        if counts[a["tab"]] <= MAX_ARTICLES_PER_TAB_STORED:
            out.append(a)
    return out


def build_feed(cfg: Dict[str, Any], previous: List[Article], getter: Callable[[str], bytes],
               now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc).replace(microsecond=0)
    settings = cfg["settings"]
    kf = KeywordFilter(cfg["keywords"])
    cutoff = now - timedelta(hours=int(settings["max_age_hours"]))
    statuses, kept_all, suspects_all = [], [], []
    # with AI on, give it a bigger pool per tab (it removes politics), and cut to `max` afterwards
    pool_factor = float(settings.get("ai_pool_factor", 2)) if (cfg.get("ai") or {}).get("enabled") else 1.0
    stats = {"fetched": 0, "too_old": 0, "blocked": 0, "allow_matched": 0}

    for tab in cfg["tabs"]:
        tab_kept, tab_suspects = [], []
        for i, src in enumerate(tab["sources"]):
            src = dict(src, _key="%s:%d" % (tab["id"], i))
            arts, st = fetch_source(src, tab["id"], settings, now, getter)
            statuses.append(st)
            stats["fetched"] += len(arts)
            for a in arts:
                if datetime.fromisoformat(a["published"]) < cutoff:
                    stats["too_old"] += 1
                    continue
                verdict, word, quiet = kf.classify(a["title"], a["summary"])
                a["quiet"] = quiet
                if verdict == "block":
                    a["blocked_by"] = word
                    tab_suspects.append(a)
                    stats["blocked"] += 1
                else:
                    if verdict == "allow":
                        stats["allow_matched"] += 1
                    tab_kept.append(a)
        clustered = cluster(tab_kept, float(settings["similarity"]))
        mark_important(clustered, int(settings["important_min_sources"]))
        kept_all.extend(rank_and_cap(clustered, tab, settings, factor=pool_factor))
        suspects_all.extend(cluster(tab_suspects, float(settings["similarity"])))

    # the same story may show up in two tabs – keep it in the first tab only (tab order = priority)
    seen_links, seen_tokens, unique = set(), [], []
    for a in kept_all:
        toks = title_tokens(a["title"])
        if a["link"] in seen_links or any(similarity(toks, t) >= 0.6 for t in seen_tokens):
            continue
        seen_links.add(a["link"])
        seen_tokens.append(toks)
        unique.append(a)

    # Only articles the AI has not reviewed yet go to it (saves quota).
    # "Reviewed" = has a score. Stored articles without one (Gemini was down last time) get a second chance.
    prev_ids = {a.get("id") for a in previous}
    fresh = [a for a in unique if a["id"] not in prev_ids]
    seen_before = [a for a in unique if a["id"] in prev_ids]
    fresh_suspects = [a for a in suspects_all if a["id"] not in prev_ids]
    ai_on = bool((cfg.get("ai") or {}).get("enabled"))
    retry_prev = [a for a in previous if ai_on and "score" not in a][:MAX_AI_RETRY]
    retry_ids = {a["id"] for a in retry_prev}

    reviewed, ai_info = ai_stage(fresh + retry_prev, fresh_suspects, cfg)
    # stories from this scan that are already stored count toward each tab's max (else the overflow
    # of the 2x AI pool would come back as "new" on every run and the tab would keep growing)
    already: Dict[str, int] = {}
    for a in seen_before:
        already[a["tab"]] = already.get(a["tab"], 0) + 1
    final = final_cap([a for a in reviewed if a["id"] not in retry_ids], cfg["tabs"], settings, already)
    if ai_info.get("used") and retry_prev:
        # re-reviewed articles replace their stored copy; ones the AI removed disappear
        redone = {a["id"]: a for a in reviewed if a["id"] in retry_ids}
        previous = [redone.get(a.get("id"), a) for a in previous
                    if a.get("id") not in retry_ids or a.get("id") in redone]
        seen_before = [a for a in seen_before if a["id"] not in retry_ids or a["id"] in redone]
        ai_info["retried_old"] = len(retry_prev)
    if any(a.get("top") for a in final):
        # a new "top 5" replaces the previous one
        previous = [{k: v for k, v in a.items() if k != "top"} for a in previous]
    merged = merge_previous(final + seen_before, previous, now, int(settings["max_age_hours"]),
                            [t["id"] for t in cfg["tabs"]])
    stats["images_filled"] = fill_images(merged, settings, getter)

    return {
        "generated_at": now.isoformat(),
        "tabs": [{"id": t["id"], "name": t["name"], "emoji": t["emoji"]} for t in cfg["tabs"]],
        "articles": merged,
        "sources": statuses,
        "ai": ai_info,
        "stats": stats,
    }


def network_getter(url: str) -> bytes:
    for attempt in range(2):
        try:
            return http_get(url).content
        except requests.RequestException:
            if attempt == 1:
                raise
            time.sleep(1.5)
    raise RuntimeError("unreachable")


def main(argv: Optional[List[str]] = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(here, "config.yaml"))
    p.add_argument("--out", default=os.path.join(here, "feed.json"))
    p.add_argument("--offline", metavar="SAMPLE", help="use a local RSS file for every source (no network)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    if args.offline:
        sample = open(args.offline, "rb").read()
        getter = lambda url: sample  # noqa: E731
    else:
        getter = network_getter

    feed = build_feed(cfg, load_previous(args.out), getter)

    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(feed, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)

    ok = sum(1 for s in feed["sources"] if s["ok"])
    log.info("sources: %d/%d ok | articles in feed: %d | %s", ok, len(feed["sources"]), len(feed["articles"]),
             json.dumps(feed["stats"], ensure_ascii=False))
    for s in feed["sources"]:
        if not s["ok"]:
            log.info("  ✗ [%s] %s – %s", s["tab"], s["name"], s["error"])
        elif s["via"] == "google" and any(src.get("url") for t in cfg["tabs"] if t["id"] == s["tab"]
                                          for src in t["sources"] if src.get("name") == s["name"]):
            log.info("  ↪ [%s] %s – RSS failed, used Google News", s["tab"], s["name"])
    # exit non-zero only if *nothing* worked (so GitHub Actions flags a real outage)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
