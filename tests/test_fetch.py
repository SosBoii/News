import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import fetch  # noqa: E402

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)


def rss(items):
    """Build a tiny RSS document. items: list of dicts (title, link, hours_ago, image, desc, source)."""
    parts = ['<?xml version="1.0" encoding="UTF-8"?><rss version="2.0" '
             'xmlns:media="http://search.yahoo.com/mrss/"><channel><title>t</title>']
    for it in items:
        when = (NOW - timedelta(hours=it.get("hours_ago", 1))).strftime("%a, %d %b %Y %H:%M:%S +0000")
        parts.append("<item><title>%s</title><link>%s</link><pubDate>%s</pubDate>" % (it["title"], it["link"], when))
        if it.get("desc"):
            parts.append("<description><![CDATA[%s]]></description>" % it["desc"])
        if it.get("image"):
            parts.append('<media:content url="%s" medium="image"/>' % it["image"])
        if it.get("source"):
            parts.append('<source url="https://x">%s</source>' % it["source"])
        parts.append("</item>")
    parts.append("</channel></rss>")
    return "".join(parts).encode("utf-8")


def make_cfg(tabs, block=(), allow=(), quiet=(), **settings):
    return {
        "settings": {**fetch.DEFAULT_SETTINGS, "fetch_images": False, **settings},
        "keywords": {"block": list(block), "allow": list(allow), "quiet": list(quiet)},
        "tabs": tabs,
        "ai": {"enabled": False},
    }


def getter_from(mapping):
    def g(url):
        for key, body in mapping.items():
            if key in url:
                if isinstance(body, Exception):
                    raise body
                return body
        raise IOError("no route for " + url)
    return g


# ---------------------------------------------------------------- keywords

class TestKeywordFilter:
    def setup_method(self):
        self.kf = fetch.KeywordFilter({
            "block": ["הליכוד", 'ח"כ', "Senate vote", "קואליציה"],
            "allow": ["טראמפ הודיע", "tariffs"],
            "quiet": ["כדורגל"],
        })

    def test_plain_block(self):
        assert self.kf.classify("סערה בהליכוד")[0] == "block"

    def test_hebrew_prefixes(self):
        assert self.kf.classify("ראש הממשלה נפגש עם בכירים והליכוד מגיב")[0] == "block"
        assert self.kf.classify("משבר בקואליציה")[0] == "block"

    def test_hebrew_gershayim_variants(self):
        assert self.kf.classify("ח״כ לשעבר נחקר")[0] == "block"   # Hebrew gershayim ״
        assert self.kf.classify('ח"כ לשעבר נחקר')[0] == "block"   # ASCII quote

    def test_no_partial_word_match(self):
        # "Senate vote" must not match "Senate voters" substrings incorrectly in other words
        assert self.kf.classify("Senatorial votes")[0] == "ok"

    def test_english_case_insensitive_and_spacing(self):
        assert self.kf.classify("The  SENATE   Vote is today")[0] == "block"

    def test_allow_wins_over_block(self):
        verdict, word, _ = self.kf.classify("טראמפ הודיע על מכסים; הליכוד מגיב")
        assert verdict == "allow" and word == "טראמפ הודיע"

    def test_summary_is_checked(self):
        assert self.kf.classify("כותרת תמימה", "בתוך הכתבה: הליכוד")[0] == "block"

    def test_quiet_flag(self):
        verdict, _, quiet = self.kf.classify("מכבי ניצחה בכדורגל")
        assert verdict == "ok" and quiet is True

    def test_ok(self):
        assert self.kf.classify("גילוי חדש בחלל") == ("ok", None, False)


# ---------------------------------------------------------------- fetching

class TestFetchSource:
    def test_rss_parsed_with_image_and_source(self):
        body = rss([{"title": "Hello", "link": "https://a.com/1", "image": "https://a.com/i.jpg"}])
        arts, st = fetch.fetch_source({"name": "A", "url": "https://a.com/rss"}, "t", fetch.DEFAULT_SETTINGS, NOW,
                                      getter_from({"a.com/rss": body}))
        assert st["ok"] and st["via"] == "rss"
        assert arts[0]["title"] == "Hello" and arts[0]["source"] == "A" and arts[0]["image"] == "https://a.com/i.jpg"

    def test_falls_back_to_google_when_rss_fails(self):
        body = rss([{"title": "כותרת - וואלה", "link": "https://news.google.com/x", "source": "וואלה"}])
        arts, st = fetch.fetch_source({"name": "וואלה", "url": "https://bad/rss", "google": "site:walla.co.il"},
                                      "t", fetch.DEFAULT_SETTINGS, NOW,
                                      getter_from({"bad/rss": IOError("404"), "news.google.com": body}))
        assert st["ok"] and st["via"] == "google"
        assert arts[0]["title"] == "כותרת"          # " - publisher" suffix removed
        assert arts[0]["source"] == "וואלה"

    def test_failure_is_reported_not_raised(self):
        arts, st = fetch.fetch_source({"name": "X", "url": "https://bad/rss"}, "t", fetch.DEFAULT_SETTINGS, NOW,
                                      getter_from({"bad": IOError("boom")}))
        assert arts == [] and not st["ok"] and "boom" in st["error"]

    def test_google_url_language(self):
        assert "ceid=IL:he" in fetch.google_news_url("חלל")
        u = fetch.google_news_url("NASA", "en", "when:2d")
        assert "ceid=US:en" in u and "when%3A2d" in u

    def test_bing_entry_has_image_real_link_and_publisher(self):
        body = ('<?xml version="1.0"?><rss version="2.0" xmlns:News="https://www.bing.com/news/search?q=x&amp;format=rss">'
                '<channel><title>Bing</title><item><title>AI beats humans at chess again</title>'
                '<link>http://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;aid=&amp;tid=1&amp;url=https%3a%2f%2fexample.com%2fai-story&amp;c=1</link>'
                '<pubDate>Mon, 19 Oct 2026 11:00:00 GMT</pubDate><description>Some text</description>'
                '<News:Source>Example Times</News:Source>'
                '<News:Image>http://www.bing.com/th?id=ONUT.abc&amp;pid=News</News:Image></item></channel></rss>').encode()
        arts, st = fetch.fetch_source({"name": "Bing News", "bing": "ai", "google": "ai", "lang": "en"}, "tech",
                                      fetch.DEFAULT_SETTINGS, NOW, getter_from({"bing.com": body}))
        assert st["via"] == "bing"
        a = arts[0]
        assert a["link"] == "https://example.com/ai-story"            # bing redirect unwrapped
        assert a["source"] == "Example Times"
        assert a["image"].startswith("https://www.bing.com/th?id=ONUT.abc") and "w=700" in a["image"]

    def test_order_rss_then_bing_then_google(self):
        bing_body = rss([{"title": "From bing", "link": "https://b/1"}])
        google_body = rss([{"title": "From google - G", "link": "https://news.google.com/x", "source": "G"}])
        src = {"name": "S", "url": "https://dead/rss", "bing": "q", "google": "q", "lang": "en"}
        arts, st = fetch.fetch_source(src, "t", fetch.DEFAULT_SETTINGS, NOW,
                                      getter_from({"dead": IOError("x"), "bing.com": bing_body, "news.google.com": google_body}))
        assert st["via"] == "bing" and arts[0]["title"] == "From bing"
        arts, st = fetch.fetch_source(src, "t", fetch.DEFAULT_SETTINGS, NOW,
                                      getter_from({"dead": IOError("x"), "bing.com": IOError("y"), "news.google.com": google_body}))
        assert st["via"] == "google" and arts[0]["title"] == "From google"

    def test_unwrap_bing_link_leaves_normal_links(self):
        assert fetch.unwrap_bing_link("https://ynet.co.il/a") == "https://ynet.co.il/a"


# ---------------------------------------------------------------- clustering / ranking

def art(title, source, hours_ago=1, link=None, tab="t", **kw):
    a = {"id": fetch.article_id(link or title + source, title), "tab": tab, "title": title, "summary": "",
         "link": link or "https://%s/%d" % (source, abs(hash(title)) % 10000), "source": source, "lang": "he",
         "image": "", "published": (NOW - timedelta(hours=hours_ago)).isoformat(), "quiet": False}
    a.update(kw)
    return a


class TestCluster:
    def test_same_story_merged_with_also(self):
        merged = fetch.cluster([
            art("רעידת אדמה חזקה הורגשה בצפון הארץ", "ynet", 3),
            art("רעידת אדמה הורגשה בצפון הארץ הבוקר", "וואלה", 2),
            art("רעידת אדמה חזקה בצפון", "מעריב", 1),
            art("מחירי הדלק יעלו בחודש הבא", "ynet", 1),
        ], 0.45)
        assert len(merged) == 2
        quake = next(a for a in merged if "רעידת" in a["title"])
        assert quake["source"] == "ynet"                       # earliest is the card
        assert sorted(quake["also"]) == sorted(["וואלה", "מעריב"])

    def test_different_stories_not_merged(self):
        merged = fetch.cluster([art("NASA launches new telescope", "BBC"), art("Stock market hits record", "BBC")], 0.45)
        assert len(merged) == 2

    def test_image_borrowed_from_duplicate(self):
        merged = fetch.cluster([art("Big storm hits coast tonight", "A", 2),
                                art("Big storm hits the coast", "B", 1, image="https://b/img.jpg")], 0.45)
        assert merged[0]["image"] == "https://b/img.jpg"


class TestRankAndCap:
    def test_per_source_and_tab_limit(self):
        arts = [art("ynet story %d unique words %d" % (i, i), "ynet", i) for i in range(5)] + \
               [art("walla story %d other %d" % (i, i), "וואלה", i) for i in range(5)]
        out = fetch.rank_and_cap(arts, {"max": 5, "per_source": 3}, fetch.DEFAULT_SETTINGS)
        assert len(out) == 5
        assert sum(1 for a in out if a["source"] == "ynet") <= 3

    def test_quiet_goes_last_and_multisource_first(self):
        a_quiet = art("football", "S", 0, quiet=True)
        a_multi = art("big", "S2", 5, multi=3)
        a_new = art("new", "S3", 0)
        out = fetch.rank_and_cap([a_quiet, a_new, a_multi], {"max": 10}, fetch.DEFAULT_SETTINGS)
        assert [a["title"] for a in out] == ["big", "new", "football"]

    def test_keywords_and_sources_never_set_important(self):
        """🔴 comes only from the AI. Many sources only raises the ranking (`multi`)."""
        a = art("x", "S", feeds=["israel:0", "israel:1", "israel:2"])
        g = art("z", "Google", also=["A", "B", "C", "D"], feeds=["world:1"])
        fetch.mark_important([a, g], 3)
        assert a["important"] is False and a["multi"] == 3
        assert g["important"] is False and g["multi"] == 1

    def test_also_list_is_capped(self):
        arts = [art("Elon Musk pledges SpaceX share every kid", "P%d" % i, 10 - i) for i in range(10)]
        merged = fetch.cluster(arts, 0.45)
        assert len(merged) == 1 and len(merged[0]["also"]) == fetch.MAX_ALSO


# ---------------------------------------------------------------- end to end

class TestBuildFeed:
    def cfg(self, **kw):
        tabs = [
            {"id": "israel", "name": "בארץ", "emoji": "🇮🇱", "max": 12, "per_source": 3, "sources": [
                {"name": "ynet", "url": "https://ynet/rss"},
                {"name": "וואלה", "url": "https://walla/rss"},
                {"name": "מעריב", "url": "https://maariv/rss"},
                {"name": "broken", "url": "https://broken/rss"},
            ]},
            {"id": "sports", "name": "ספורט", "emoji": "⚽", "max": 3, "sources": [
                {"name": "sport", "url": "https://sport/rss"}]},
        ]
        return make_cfg(tabs, block=["הליכוד"], allow=["טראמפ הודיע"], quiet=["כדורגל"], **kw)

    def getter(self):
        return getter_from({
            "ynet/rss": rss([
                {"title": "שריפה גדולה פרצה ביער ירושלים", "link": "https://ynet/1", "hours_ago": 3},
                {"title": "סערה בהליכוד אחרי ההצבעה", "link": "https://ynet/2"},
                {"title": "טראמפ הודיע על מכסים חדשים, הליכוד מגיב", "link": "https://ynet/3"},
                {"title": "כתבה ישנה מאוד", "link": "https://ynet/old", "hours_ago": 200},
                {"title": "סיפור רביעי של ynet על מזג האוויר", "link": "https://ynet/4"},
                {"title": "סיפור חמישי של ynet על תחבורה ציבורית", "link": "https://ynet/5"},
                {"title": "סיפור שישי של ynet על חינוך", "link": "https://ynet/6"},
            ]),
            "walla/rss": rss([{"title": "שריפה גדולה ביער ירושלים, כוחות רבים בשטח", "link": "https://walla/1", "hours_ago": 2}]),
            "maariv/rss": rss([{"title": "שריפה ביער ירושלים: כוחות כיבוי רבים", "link": "https://maariv/1", "hours_ago": 1}]),
            "broken/rss": IOError("connection refused"),
            "sport/rss": rss([{"title": t, "link": "https://sport/%d" % i} for i, t in enumerate([
                "מכבי חיפה ניצחה בכדורגל", "הפועל באר שבע החתימה שוער", "ליברפול הפסידה בבית",
                "לברון ג'יימס שבר שיא", "בית\"ר ירושלים מחליפה מאמן", "גמר גביע המדינה נקבע",
            ])]),
        })

    def test_pipeline(self):
        feed = fetch.build_feed(self.cfg(), [], self.getter(), NOW)
        arts = feed["articles"]
        titles = [a["title"] for a in arts]

        assert "סערה בהליכוד אחרי ההצבעה" not in titles               # blocked
        assert any(t.startswith("טראמפ הודיע") for t in titles)        # allow beats block
        assert "כתבה ישנה מאוד" not in titles                          # too old
        fire = [a for a in arts if "שריפה" in a["title"]]
        assert len(fire) == 1 and len(fire[0]["also"]) == 2              # merged across 3 sites
        assert len(fire[0]["feeds"]) == 3
        assert fire[0]["important"] is False                             # no AI => never 🔴
        assert fire[0]["multi"] == 3
        assert sum(1 for a in arts if a["tab"] == "israel" and a["source"] == "ynet") <= 3
        assert sum(1 for a in arts if a["tab"] == "sports") == 3         # sports capped at 3

        broken = next(s for s in feed["sources"] if s["name"] == "broken")
        assert not broken["ok"] and "refused" in broken["error"]
        assert feed["ai"]["used"] is False
        assert [t["id"] for t in feed["tabs"]] == ["israel", "sports"]
        json.dumps(feed, ensure_ascii=False)                             # serialisable

    def test_second_run_keeps_old_and_does_not_repeat(self):
        first = fetch.build_feed(self.cfg(), [], self.getter(), NOW)
        later = NOW + timedelta(hours=12)
        second = fetch.build_feed(self.cfg(), first["articles"], self.getter(), later)
        ids1 = sorted(a["id"] for a in first["articles"])
        ids2 = sorted(a["id"] for a in second["articles"])
        assert ids1 == ids2                                              # nothing duplicated
        fire = next(a for a in second["articles"] if "שריפה" in a["title"])
        assert fire["first_seen"] == NOW.isoformat()                     # keeps original first_seen

    def test_expired_articles_dropped_from_previous(self):
        old = art("ישן", "ynet", 1, tab="israel", first_seen=(NOW - timedelta(hours=100)).isoformat())
        feed = fetch.build_feed(self.cfg(), [old], self.getter(), NOW)
        assert all(a["title"] != "ישן" for a in feed["articles"])

    def test_removed_tab_articles_dropped(self):
        stale = art("מטאב שנמחק", "x", 1, tab="deleted_tab", first_seen=NOW.isoformat())
        feed = fetch.build_feed(self.cfg(), [stale], self.getter(), NOW)
        assert all(a["tab"] != "deleted_tab" for a in feed["articles"])

    def test_all_sources_down_still_produces_feed(self):
        feed = fetch.build_feed(self.cfg(), [], getter_from({"": IOError("offline")}), NOW)
        assert feed["articles"] == [] and all(not s["ok"] for s in feed["sources"])


# ---------------------------------------------------------------- images & config

def test_find_og_image_both_attribute_orders():
    assert fetch.find_og_image('<meta property="og:image" content="https://x/a.jpg">') == "https://x/a.jpg"
    assert fetch.find_og_image("<meta content='https://x/b.jpg' property='og:image' />") == "https://x/b.jpg"
    assert fetch.find_og_image("<html></html>") == ""


def test_fill_images_respects_budget_and_skips_google():
    arts = [art("a", "s", link="https://site/1"), art("b", "s", link="https://news.google.com/x"),
            art("c", "s", link="https://site/2")]
    page = b'<meta property="og:image" content="https://site/img.jpg">'
    n = fetch.fill_images(arts, {"fetch_images": True, "max_image_fetches": 1}, lambda u: page)
    assert n == 1 and arts[0]["image"] and not arts[1]["image"] and not arts[2]["image"]


def test_real_config_loads():
    cfg = fetch.load_config(os.path.join(os.path.dirname(HERE), "config.yaml"))
    ids = [t["id"] for t in cfg["tabs"]]
    assert "israel" in ids and "sports" in ids
    israel = next(t for t in cfg["tabs"] if t["id"] == "israel")
    assert len(israel["sources"]) >= 5
    sports = next(t for t in cfg["tabs"] if t["id"] == "sports")
    assert sports["max"] <= 3
    assert "כדורגל" in cfg["keywords"]["quiet"]
    assert "בחירות" not in cfg["keywords"]["block"]


def test_duplicate_tab_id_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("tabs:\n  - id: a\n  - id: a\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fetch.load_config(str(p))
