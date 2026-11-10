import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import ai  # noqa: E402
import fetch  # noqa: E402

CFG = {"ai": {"enabled": True, "model": "gemini-test", "instructions": "בלי פוליטיקה"}}


def a(title, quiet=False, **kw):
    d = {"id": title, "title": title, "summary": "", "source": "S", "published": "2026-10-19T10:00:00+00:00",
         "quiet": quiet}
    d.update(kw)
    return d


def gemini_reply(items):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(items, ensure_ascii=False)}]}}]}


def prompt_items(prompt):
    """The article list is the JSON array after the last 'הכתבות:' marker."""
    return json.loads(prompt.rsplit("הכתבות:", 1)[1].strip())


class FakePost:
    """Answers each batch using a function title -> verdict dict."""

    def __init__(self, decide, fail=None):
        self.decide, self.fail, self.calls = decide, fail, []

    def __call__(self, url, headers, body):
        self.calls.append((url, headers, body))
        if self.fail:
            raise self.fail
        items = prompt_items(body["contents"][0]["parts"][0]["text"])
        return gemini_reply([dict(self.decide(it), i=it["i"]) for it in items])


def test_disabled_or_no_key_is_passthrough():
    kept = [a("x")]
    out, info = ai.run(kept, [a("pol")], {"ai": {"enabled": False}}, post=FakePost(lambda it: {}), api_key="k")
    assert out == kept and info["used"] is False
    out, info = ai.run(kept, [a("pol")], CFG, post=FakePost(lambda it: {}), api_key="")
    assert out == kept and info["reason"] == "no GEMINI_API_KEY"


def test_filter_summary_score_and_top():
    def decide(it):
        t = it["title"]
        if t == "politics":
            return {"keep": False}
        if t == "quake":
            return {"keep": True, "important": True, "score": 5, "summary": "רעידת אדמה חזקה"}
        return {"keep": True, "important": False, "score": 3, "summary": "סיכום " + t}

    kept = [a("politics"), a("quake"), a("tech"), a("football", quiet=True)]
    out, info = ai.run(kept, [], CFG, post=FakePost(decide), api_key="k")
    titles = [x["title"] for x in out]
    assert "politics" not in titles and info["removed"] == 1
    quake = next(x for x in out if x["title"] == "quake")
    assert quake["ai_summary"] == "רעידת אדמה חזקה" and quake["important"] and quake["top"] == 1
    football = next(x for x in out if x["title"] == "football")
    assert football["score"] <= 2 and "top" not in football       # quiet never in top
    assert info["used"] and info["summarized"] == 3


def test_suspect_rescued_only_if_keep_and_important():
    def decide(it):
        if it["title"] == "big security event":
            return {"keep": True, "important": True, "score": 5, "summary": "אירוע ביטחוני"}
        return {"keep": True, "important": False, "score": 3, "summary": "s"}

    out, info = ai.run([], [a("big security event"), a("party fight")], CFG, post=FakePost(decide), api_key="k")
    assert [x["title"] for x in out] == ["big security event"] and info["rescued"] == 1


def test_suspect_flag_sent_to_model():
    fp = FakePost(lambda it: {"keep": False})
    ai.run([a("ok")], [a("pol")], CFG, post=fp, api_key="k")
    items = prompt_items(fp.calls[0][2]["contents"][0]["parts"][0]["text"])
    assert items[0].get("suspect") is None and items[1]["suspect"] is True


def test_key_sent_in_header_not_url():
    fp = FakePost(lambda it: {"keep": True})
    ai.run([a("x")], [], CFG, post=fp, api_key="SECRET123")
    url, headers, _ = fp.calls[0]
    assert "SECRET123" not in url and headers["x-goog-api-key"] == "SECRET123"


def test_quota_error_falls_back_to_keywords():
    kept, suspects = [a("x"), a("y")], [a("pol")]
    out, info = ai.run(kept, suspects, CFG, post=FakePost(None, fail=ai.QuotaExceeded()), api_key="k")
    assert [x["title"] for x in out] == ["x", "y"]                 # kept stay, suspects dropped
    assert info["used"] is False and "quota" in info["reason"]


def test_bad_json_falls_back():
    def bad_post(url, headers, body):
        return {"candidates": [{"content": {"parts": [{"text": "sorry, I can't"}]}}]}
    out, info = ai.run([a("x")], [a("pol")], CFG, post=bad_post, api_key="k")
    assert [x["title"] for x in out] == ["x"] and info["failed_batches"] == 1


def test_markdown_fenced_json_is_parsed():
    def fenced(url, headers, body):
        txt = "```json\n" + json.dumps([{"i": 0, "keep": True, "score": 4, "summary": "יופי"}]) + "\n```"
        return {"candidates": [{"content": {"parts": [{"text": txt}]}}]}
    out, _ = ai.run([a("x")], [], CFG, post=fenced, api_key="k")
    assert out[0]["ai_summary"] == "יופי"


def test_batches_of_40():
    fp = FakePost(lambda it: {"keep": True, "score": 3})
    ai.run([a("t%d" % i) for i in range(85)], [], CFG, post=fp, api_key="k")
    assert len(fp.calls) == 3


def test_build_feed_sends_only_new_articles_to_ai(monkeypatch):
    sent = []

    def fake_stage(kept, suspects, cfg):
        sent.append([x["title"] for x in kept])
        return [dict(x, ai_summary="s", score=3) for x in kept], {"used": True}   # real AI always sets score

    monkeypatch.setattr(fetch, "ai_stage", fake_stage)
    body = (b'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>'
            b'<item><title>Rocket launch today from Florida</title><link>https://s/1</link>'
            b'<pubDate>Mon, 19 Oct 2026 10:00:00 +0000</pubDate></item></channel></rss>')
    cfg = {"settings": dict(fetch.DEFAULT_SETTINGS, fetch_images=False),
           "keywords": {"block": [], "allow": [], "quiet": []},
           "tabs": [{"id": "sci", "name": "s", "emoji": "x", "max": 5, "sources": [{"name": "S", "url": "https://s/rss"}]}],
           "ai": {"enabled": True}}
    now = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)
    first = fetch.build_feed(cfg, [], lambda u: body, now)
    second = fetch.build_feed(cfg, first["articles"], lambda u: body, now)
    assert sent == [["Rocket launch today from Florida"], []]
    assert second["articles"][0]["ai_summary"] == "s"              # summary kept from first run


def test_ai_gets_bigger_pool_and_tab_is_cut_after(monkeypatch):
    """With AI on, 2x max goes to the AI; after it removes some, the tab is still filled up to max."""
    sent = []

    def fake_stage(kept, suspects, cfg):
        sent.extend(kept)
        out = [dict(x, score=5 if x["title"].startswith("Good") else 2) for x in kept if "politics" not in x["title"]]
        return out, {"used": True}

    monkeypatch.setattr(fetch, "ai_stage", fake_stage)
    titles = ["Senate politics deadlock continues", "Mayor politics scandal widens", "Party politics primaries begin",
              "Good rocket launch reaches orbit", "Cheese festival opens downtown", "Good volcano erupts near Iceland",
              "Trains delayed after heavy rain", "Bees return to city rooftops"]
    items = "".join('<item><title>%s</title><link>https://s/%d</link>'
                    '<pubDate>Mon, 19 Oct 2026 %02d:00:00 +0000</pubDate></item>' % (t, i, 10 - i // 2)
                    for i, t in enumerate(titles))
    body = ('<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>%s</channel></rss>' % items).encode()
    cfg = {"settings": dict(fetch.DEFAULT_SETTINGS, fetch_images=False, per_source=10),
           "keywords": {"block": [], "allow": [], "quiet": []},
           "tabs": [{"id": "x", "name": "x", "emoji": "x", "max": 4, "sources": [{"name": "S", "url": "https://s/rss"}]}],
           "ai": {"enabled": True}}
    feed = fetch.build_feed(cfg, [], lambda u: body, datetime(2026, 10, 19, 12, tzinfo=timezone.utc))
    assert len(sent) == 8                                           # 2 x max
    got = [a["title"] for a in feed["articles"]]
    assert len(got) == 4 and not any("politics" in t for t in got)  # tab still full after AI removed 3
    assert {"Good rocket launch reaches orbit", "Good volcano erupts near Iceland"} <= set(got)  # higher score wins
    assert sorted(a["top"] for a in feed["articles"] if a.get("top")) == [1, 2, 3, 4]


def _rss(titles, hour=10):
    items = "".join('<item><title>%s</title><link>https://s/%d</link>'
                    '<pubDate>Mon, 19 Oct 2026 %02d:00:00 +0000</pubDate></item>' % (t, i, hour) for i, t in enumerate(titles))
    return ('<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>%s</channel></rss>' % items).encode()


def _one_tab_cfg(mx):
    return {"settings": dict(fetch.DEFAULT_SETTINGS, fetch_images=False, per_source=50),
            "keywords": {"block": [], "allow": [], "quiet": []},
            "tabs": [{"id": "x", "name": "x", "emoji": "x", "max": mx, "sources": [{"name": "S", "url": "https://s/rss"}]}],
            "ai": {"enabled": True}}


TITLES = ["Rocket launch reaches orbit", "Cheese festival opens downtown", "Volcano erupts near Iceland",
          "Trains delayed after heavy rain", "Bees return to city rooftops", "Museum unveils ancient mosaic"]


def test_articles_stored_without_ai_are_reviewed_next_run(monkeypatch):
    now = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)
    body = _rss(TITLES[:3])
    cfg = _one_tab_cfg(5)
    monkeypatch.setattr(fetch, "ai_stage", lambda k, s, c: (k, {"used": False}))      # Gemini down
    first = fetch.build_feed(cfg, [], lambda u: body, now)
    assert len(first["articles"]) == 3 and not any("score" in a for a in first["articles"])

    sent = []

    def up(kept, suspects, c):                                                       # Gemini back; drops cheese
        sent.extend(x["title"] for x in kept)
        return [dict(x, score=4, ai_summary="s") for x in kept if "Cheese" not in x["title"]], {"used": True}
    monkeypatch.setattr(fetch, "ai_stage", up)
    second = fetch.build_feed(cfg, first["articles"], lambda u: body, now)
    assert sorted(sent) == sorted(TITLES[:3])                                        # old ones re-sent
    titles = sorted(a["title"] for a in second["articles"])
    assert titles == sorted([TITLES[0], TITLES[2]])                                  # AI-removed one is gone
    assert all(a.get("ai_summary") == "s" for a in second["articles"])
    assert second["ai"]["retried_old"] == 3


def test_repeat_runs_do_not_grow_tab_beyond_max(monkeypatch):
    """2x pool goes to the AI; the overflow must not re-enter as 'new' on the next run."""
    now = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)
    body = _rss(TITLES)                               # 6 articles, max 3 -> pool of 6
    cfg = _one_tab_cfg(3)
    monkeypatch.setattr(fetch, "ai_stage", lambda k, s, c: ([dict(x, score=3) for x in k], {"used": True}))
    first = fetch.build_feed(cfg, [], lambda u: body, now)
    second = fetch.build_feed(cfg, first["articles"], lambda u: body, now)
    assert len(first["articles"]) == 3 and len(second["articles"]) == 3


def test_timeout_falls_back_to_next_model(monkeypatch):
    def boom(*a, **k):
        raise ai.requests.Timeout("slow")
    monkeypatch.setattr(ai.requests, "post", boom)
    try:
        ai._default_post("u", {}, {})
        assert False, "should raise"
    except ai.ModelUnavailable:
        pass


def test_503_is_retried_on_same_model(monkeypatch):
    class R:
        def __init__(self, code):
            self.status_code, self.text = code, ""

        def json(self):
            return {"ok": 1}
    codes = [503, 503, 200]
    monkeypatch.setattr(ai.requests, "post", lambda *a, **k: R(codes.pop(0)))
    monkeypatch.setattr(ai.time, "sleep", lambda s: None)
    assert ai._default_post("u", {}, {}) == {"ok": 1} and codes == []
    codes[:] = [503, 503, 503, 503]
    try:
        ai._default_post("u", {}, {})
        assert False, "should raise"
    except ai.ModelUnavailable:
        pass


def test_falls_back_to_next_model_when_unavailable():
    calls = []

    def post(url, headers, body):
        calls.append(url)
        if "/big:" in url:
            raise ai.ModelUnavailable("HTTP 503")
        items = prompt_items(body["contents"][0]["parts"][0]["text"])
        txt = json.dumps([{"i": it["i"], "keep": True, "score": 3, "summary": "ok"} for it in items])
        return {"candidates": [{"content": {"parts": [{"text": txt}]}}]}

    cfg = {"ai": {"enabled": True, "models": ["big", "lite"]}}
    out, info = ai.run([a("t%d" % i) for i in range(45)], [], cfg, post=post, api_key="k")
    assert info["model"] == "lite" and info["skipped_models"] == ["big: ModelUnavailable"]
    assert ["/big:" in u for u in calls] == [True, False, False]   # overloaded model dropped for the rest of the run
    assert all(x.get("ai_summary") == "ok" for x in out)


def test_all_models_unavailable_keeps_feed():
    def post(url, headers, body):
        raise ai.ModelUnavailable("HTTP 404")
    out, info = ai.run([a("x")], [a("pol")], {"ai": {"enabled": True, "models": ["m1", "m2"]}}, post=post, api_key="k")
    assert [x["title"] for x in out] == ["x"] and info["used"] is False


def test_real_config_uses_model_list():
    cfg = fetch.load_config(os.path.join(os.path.dirname(HERE), "config.yaml"))
    assert ai._model_list(cfg["ai"])[0].startswith("gemini-")


def test_load_api_key_env_wins_then_file(tmp_path, monkeypatch):
    f = tmp_path / "gemini_key.txt"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert ai.load_api_key(str(f)) == ""                           # no env, no file
    f.write_text("# paste below\n\n  AIzaFILE  \n", encoding="utf-8")
    assert ai.load_api_key(str(f)) == "AIzaFILE"
    f.write_text("GEMINI_API_KEY=AIzaKV\n", encoding="utf-8")
    assert ai.load_api_key(str(f)) == "AIzaKV"
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaENV")
    assert ai.load_api_key(str(f)) == "AIzaENV"


def test_key_file_is_gitignored():
    with open(os.path.join(os.path.dirname(HERE), ".gitignore"), encoding="utf-8") as fh:
        assert "gemini_key.txt" in fh.read().split()
