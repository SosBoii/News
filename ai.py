"""
AI stage (Gemini): smart political filter, one-line Hebrew summary, importance score.

Called from fetch.ai_stage(). Contract:
    run(kept, suspects, cfg, post=None) -> (articles, info)

  kept      articles that passed keyword filtering. Without AI they all stay.
  suspects  articles that matched a blocked word. They return ONLY if the AI says keep + important.

If anything goes wrong (no key, quota, network, bad JSON) the affected articles fall back to
keyword-only behaviour: kept stay without summary, suspects are dropped. The run never fails.

The API key is read from the GEMINI_API_KEY environment variable (GitHub Secret). For local runs it may
instead be put in gemini_key.txt next to this file (gitignored). It is never logged.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("news-feed.ai")

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# "-latest" aliases follow Google's current model, so the feed survives model retirements.
DEFAULT_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
BATCH_SIZE = 40
TIMEOUT = 90
RETRY_WAITS = (5, 15)   # seconds between retries on 503; then fall back to the next model (cheaper than waiting)
TOP_N = 5

Article = Dict[str, Any]
PostFn = Callable[[str, Dict[str, str], Dict[str, Any]], Dict[str, Any]]


KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_key.txt")


class QuotaExceeded(Exception):
    pass


class ModelUnavailable(Exception):
    """Model retired (404) or overloaded (5xx) – try the next model in the list."""


def load_api_key(key_file: str = KEY_FILE) -> str:
    """Env var GEMINI_API_KEY first (GitHub Actions); else the local gitignored gemini_key.txt. Never logged."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        with open(key_file, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line.split("=", 1)[1].strip() if line.startswith("GEMINI_API_KEY=") else line
    except OSError:
        pass
    return ""


def _send(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> requests.Response:
    try:
        return requests.post(url, headers=headers, json=body, timeout=TIMEOUT)
    except (requests.Timeout, requests.ConnectionError) as e:  # slow/overloaded model – try the next one
        raise ModelUnavailable(type(e).__name__)


def _default_post(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
    r = _send(url, headers, body)
    # "high demand" (503) is usually momentary – retry the same model a few times before falling back
    for wait in RETRY_WAITS:
        if r.status_code != 503:
            break
        log.info("Gemini 503 (busy) – retrying in %ds", wait)
        time.sleep(wait)
        r = _send(url, headers, body)
    if r.status_code == 429:
        raise QuotaExceeded("rate limit / quota")
    if r.status_code == 404 or r.status_code >= 500:
        raise ModelUnavailable("HTTP %d" % r.status_code)
    if r.status_code >= 400:
        # body may echo the request; keep it short and never include headers (the key lives there)
        raise RuntimeError("HTTP %d: %s" % (r.status_code, r.text[:200]))
    return r.json()


PROMPT_HEADER = """אתה עורך חדשות אישי. המשתמש לא צורך חדשות בכלל ורוצה פיד קצר, נקי ומעניין.

ההוראות של המשתמש:
---
{instructions}
---

תקבל רשימת כתבות בפורמט JSON. לכל כתבה יש מספר i.
כתבה עם "suspect": true הכילה מילה פוליטית שנחסמה. השאר אותה רק אם היא באמת חשובה מאוד לפי סעיף 4.

החזר אך ורק מערך JSON, איבר אחד לכל כתבה, באותו סדר:
[{{"i": <מספר>, "keep": true/false, "important": true/false, "score": 1-5, "summary": "<משפט אחד בעברית, עד 25 מילים>"}}]

- keep: האם להשאיר בפיד לפי ההוראות.
- important: כמעט תמיד false. true רק לפי הכללים הקשוחים בהוראות (בערך 0–2 מכל הרשימה). בספק – false.
- score: עד כמה זה מעניין/חשוב לאדם רגיל (5 = הכי). ספורט, רכילות ובידור – לכל היותר 2.
- summary: מה קרה, בעברית פשוטה, גם אם הכתבה באנגלית. בלי "הכתבה עוסקת ב". אם keep=false אפשר מחרוזת ריקה.
- JSON תקין בלבד: בתוך טקסט אל תשתמש במירכאות " – לראשי תיבות כתוב ״ (למשל צה״ל, ארה״ב).

הכתבות:
"""


def _build_prompt(instructions: str, batch: List[Tuple[int, Article, bool]]) -> str:
    items = []
    for i, a, suspect in batch:
        item = {"i": i, "title": a["title"], "source": a.get("source", "")}
        if a.get("summary"):
            item["text"] = a["summary"][:300]
        if suspect:
            item["suspect"] = True
        items.append(item)
    return PROMPT_HEADER.format(instructions=instructions.strip()) + json.dumps(items, ensure_ascii=False)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def _parse_response(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected response shape (blocked or empty)")
    text = _FENCE.sub("", text.strip())
    data = json.loads(text)
    if isinstance(data, dict):  # sometimes wrapped: {"items": [...]}
        data = next((v for v in data.values() if isinstance(v, list)), [])
    if not isinstance(data, list):
        raise ValueError("response is not a list")
    return [d for d in data if isinstance(d, dict) and isinstance(d.get("i"), int)]


def _model_list(ai_cfg: Dict[str, Any]) -> List[str]:
    models = ai_cfg.get("models") or ai_cfg.get("model") or DEFAULT_MODELS
    if isinstance(models, str):
        models = [models]
    return [str(m).strip() for m in models if str(m).strip()]


def _post_with_fallback(post: PostFn, models: List[str], headers: Dict[str, str], body: Dict[str, Any],
                        info: Dict[str, Any]) -> Dict[str, Any]:
    """Try models in order. A model that is retired/overloaded/out of quota is dropped for the rest of the run."""
    while models:
        model = models[0]
        try:
            resp = post(API_URL.format(model=model), headers, body)
            info["model"] = model
            return resp
        except (QuotaExceeded, ModelUnavailable) as e:
            log.warning("Gemini model %s unavailable (%s) – trying next", model, e)
            info["skipped_models"].append("%s: %s" % (model, type(e).__name__))
            models.pop(0)
    raise QuotaExceeded("no Gemini model available")


def _clean_summary(s: Any) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s[:280]


def run(kept: List[Article], suspects: List[Article], cfg: Dict[str, Any],
        post: Optional[PostFn] = None, api_key: Optional[str] = None) -> Tuple[List[Article], Dict[str, Any]]:
    ai_cfg = cfg.get("ai") or {}
    models = _model_list(ai_cfg)
    info: Dict[str, Any] = {"used": False, "model": models[0] if models else None, "skipped_models": [],
                            "batches": 0, "failed_batches": 0,
                            "removed": 0, "rescued": 0, "summarized": 0, "reason": None}
    if not ai_cfg.get("enabled", False):
        info["reason"] = "disabled in config"
        return kept, info
    api_key = api_key if api_key is not None else load_api_key()
    if not api_key:
        info["reason"] = "no GEMINI_API_KEY"
        return kept, info
    if not kept and not suspects:
        info["reason"] = "nothing to process"
        return kept, info

    post = post or _default_post
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    instructions = ai_cfg.get("instructions") or ""

    entries: List[Tuple[int, Article, bool]] = [(i, a, False) for i, a in enumerate(kept)]
    entries += [(len(kept) + j, a, True) for j, a in enumerate(suspects)]
    verdicts: Dict[int, Dict[str, Any]] = {}

    for start in range(0, len(entries), BATCH_SIZE):
        batch = entries[start:start + BATCH_SIZE]
        body = {
            "contents": [{"role": "user", "parts": [{"text": _build_prompt(instructions, batch)}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        info["batches"] += 1
        try:
            try:
                parsed = _parse_response(_post_with_fallback(post, models, headers, body, info))
            except (ValueError, KeyError) as e:  # malformed JSON (e.g. a stray " in Hebrew) – ask once more
                log.warning("Gemini returned bad JSON (%s) – retrying batch once", type(e).__name__)
                parsed = _parse_response(_post_with_fallback(post, models, headers, body, info))
            for v in parsed:
                verdicts[v["i"]] = v
        except QuotaExceeded:
            info["failed_batches"] += len(range(start, len(entries), BATCH_SIZE))
            info["reason"] = "no model available (quota/overload) – rest without AI"
            log.warning("No Gemini model available; continuing with keyword filter only")
            break
        except Exception as e:  # network, bad JSON, safety block – never fail the run
            info["failed_batches"] += 1
            info["reason"] = "batch error: %s" % type(e).__name__
            log.warning("Gemini batch failed (%s) – those articles keep keyword-only result", type(e).__name__)

    info["used"] = bool(verdicts)
    out: List[Article] = []
    for i, a, suspect in entries:
        v = verdicts.get(i)
        if v is None:                      # no AI answer for this article -> keyword-only behaviour
            if not suspect:
                out.append(a)
            continue
        keep = bool(v.get("keep", True))
        important = bool(v.get("important", False)) and not a.get("quiet")
        if suspect and not (keep and important):
            continue
        if not keep:
            info["removed"] += 1
            info.setdefault("removed_titles", []).append(a.get("title", "")[:120])
            continue
        a = dict(a)
        summary = _clean_summary(v.get("summary"))
        if summary:
            a["ai_summary"] = summary
            info["summarized"] += 1
        try:
            a["score"] = max(1, min(5, int(v.get("score", 3))))
        except (TypeError, ValueError):
            a["score"] = 3
        if a.get("quiet"):
            a["score"] = min(a["score"], 2)
        if important:
            a["important"] = True
            a["important_reason"] = "ai"
        if suspect:
            info["rescued"] += 1
        out.append(a)

    # "5 things that matter today": highest score among AI-reviewed, non-quiet articles
    ranked = sorted((a for a in out if "score" in a and not a.get("quiet")),
                    key=lambda a: (not a.get("important"), -a["score"], a.get("published", "")))
    for rank, a in enumerate(ranked[:TOP_N]):
        a["top"] = rank + 1
    return out, info
