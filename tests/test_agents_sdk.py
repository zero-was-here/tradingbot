"""Request/response shapes, tool schemas, records, usage accounting, journal and prompts.

The real-SDK tests drive ``anthropic.Anthropic`` through an in-process ``httpx2.MockTransport``
(no network): they prove that the kwargs the desk builds are accepted and serialised by
the SDK, and that real SDK response objects flow through the loop.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from aurum.agents import (
    AgentModelConfig,
    CycleJournal,
    Decision,
    DeskConfig,
    Memo,
    RecordValidationError,
    StaticDeskDataProvider,
    TradingDesk,
    UsageLedger,
)
from aurum.agents.client import build_request, sanitize_assistant_content
from aurum.agents.config import DEFAULT_PRICES
from aurum.agents.prompts import (
    CHIEF_SYSTEM_PROMPT,
    ROLE_CHARTERS,
    SPECIALIST_BASE_PROMPT,
    chief_brief,
    specialist_system_prompt,
)
from aurum.agents.testing import (
    FakeAnthropicClient,
    FakeFallbackBlock,
    FakeFallbackInfo,
    FakeIterationUsage,
    FakeMessage,
    FakeUsage,
    decision_call,
    memo_call,
    message,
    refusal,
    text,
    thinking,
    tool_use,
    validate_request,
)
from aurum.agents.tools import (
    DATA_TOOL_NAMES,
    chief_tool_definitions,
    specialist_tool_definitions,
    validate_against_schema,
)
from aurum.agents.usage import estimate_response_cost

NOW = pd.Timestamp("2026-09-25 14:00", tz="UTC")


# ----------------------------------------------------------------------------- real SDK
def test_fake_messages_parse_as_real_sdk_models():
    anthropic_types = pytest.importorskip("anthropic.types.beta")
    samples = [
        message(text("hi"), tool_use("get_positions")),
        refusal(category="cyber", explanation="x"),
        FakeMessage(content=[FakeFallbackBlock(from_=FakeFallbackInfo("claude-opus-5"),
                                               to=FakeFallbackInfo("claude-opus-4-8")), text("ok")],
                    stop_reason="end_turn", model="claude-opus-4-8",
                    usage=FakeUsage(iterations=[FakeIterationUsage("fallback_message", 10, 5, "claude-opus-4-8",
                                                                   0, 0)])),
    ]
    for m in samples:
        parsed = anthropic_types.BetaMessage.model_validate(m.to_api_dict())
        assert parsed.stop_reason == m.stop_reason
        assert [b.type for b in parsed.content] == [b.type for b in m.content]


def _mock_anthropic(fake: FakeAnthropicClient, seen: list):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")

    def handler(request):
        body = json.loads(request.content)
        seen.append((dict(request.headers), body))
        betas = request.headers.get("anthropic-beta")
        if betas:
            body["betas"] = betas.split(",")
        resp = fake.beta.messages.create(**body)
        return httpx2.Response(200, json=resp.to_api_dict())

    return anthropic.Anthropic(api_key="sk-test-not-used", max_retries=0,
                               http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))


def test_full_cycle_through_real_sdk_with_mock_transport(tmp_path):
    fake = FakeAnthropicClient({
        "chief": [
            message(tool_use("get_market_snapshot"),
                    tool_use("consult_specialist", {"role": "risk_officer", "question": "Max exposure?"})),
            FakeMessage(
                content=[thinking(), tool_use("get_positions"),
                         FakeFallbackBlock(from_=FakeFallbackInfo("claude-opus-5"),
                                           to=FakeFallbackInfo("claude-opus-4-8")),
                         thinking(), text("Reviewing."), tool_use("get_calendar")],
                stop_reason="tool_use", model="claude-opus-4-8",
                usage=FakeUsage(iterations=[
                    FakeIterationUsage("message", 0, 0, "claude-opus-5", 0, 0),
                    FakeIterationUsage("fallback_message", 1000, 200, "claude-opus-4-8", 0, 0),
                ])),
            message(decision_call(action="scale", scale=0.5)),
        ],
        "risk_officer": [message(tool_use("get_risk_status")), message(memo_call(stance="neutral"))],
    })
    seen: list = []
    client = _mock_anthropic(fake, seen)
    provider = StaticDeskDataProvider({"market": {"close": 2600.0}, "risk": {"dd": -0.01}})
    desk = TradingDesk(provider, client=client, journal_dir=tmp_path)
    res = desk.run_cycle(NOW, 0.6)

    assert fake.errors == []
    assert res.status == "decided" and res.final_forecast == pytest.approx(0.3)
    assert res.memos[0].ok
    headers, body = seen[0]
    assert headers["anthropic-beta"] == "server-side-fallback-2026-07-01"
    assert body["fallbacks"] == "default"
    assert body["thinking"] == {"type": "adaptive"} and body["output_config"] == {"effort": "high"}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert all(t["strict"] is True for t in body["tools"])
    # SDK objects were echoed back verbatim (thinking signature preserved) with paired results
    chief_bodies = [b for _, b in seen if b["messages"][0]["content"][0]["text"].startswith("Agent: chief")]
    second = chief_bodies[1]["messages"]
    assert second[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "fake-signature"}
    # after the fallback boundary only post-boundary blocks (+ pre-boundary text) are echoed
    third = chief_bodies[2]["messages"]
    assert [b["type"] for b in third[3]["content"]] == ["thinking", "text", "tool_use"]
    assert third[3]["content"][2]["name"] == "get_calendar"
    assert [b["type"] for b in third[4]["content"]] == ["tool_result"]
    # cost from usage.iterations of real SDK usage objects: 5 calls x $0.01
    assert res.usage["total"]["cost_usd"] == pytest.approx(0.05)


# ----------------------------------------------------------------------------- request build
def test_build_request_variants():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Agent: chief"}]}]
    cfg = AgentModelConfig(effort="low", fallbacks=False, thinking=None)
    kw = build_request(cfg=cfg, system_prompt="S", tools=[], messages=msgs, prompt_caching=False)
    assert "betas" not in kw and "fallbacks" not in kw and "thinking" not in kw
    assert "cache_control" not in kw and "cache_control" not in kw["system"][0]
    assert kw["output_config"] == {"effort": "low"}
    kw = build_request(cfg=AgentModelConfig(), system_prompt="S", tools=[], messages=msgs, cache_ttl="1h")
    assert kw["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert validate_request(kw) == []


def test_config_validation():
    with pytest.raises(ValueError):
        AgentModelConfig(effort="turbo")
    with pytest.raises(ValueError):
        AgentModelConfig(thinking={"type": "disabled"}, effort="xhigh")
    AgentModelConfig(thinking={"type": "disabled"}, effort="high")
    with pytest.raises(ValueError):
        DeskConfig(cache_ttl="10m")
    cfg = DeskConfig(role_models={"risk_officer": AgentModelConfig(model="claude-sonnet-5", effort="low")})
    assert cfg.model_for_role("risk_officer").model == "claude-sonnet-5"
    assert cfg.model_for_role("macro_strategist").effort == "medium"
    assert cfg.model_for_role("chief").effort == "high"
    assert cfg.chief.max_tokens == 16000 and cfg.chief.model == "claude-opus-5"


def test_fallbacks_can_be_disabled_per_role():
    fake = FakeAnthropicClient({"chief": [message(decision_call())]})
    cfg = DeskConfig(chief=AgentModelConfig(effort="high", fallbacks=False))
    TradingDesk(StaticDeskDataProvider(), client=fake, config=cfg, journal_dir=None).run_cycle(NOW, 0.1)
    kw = fake.calls[0].kwargs
    assert "fallbacks" not in kw and "betas" not in kw


# ----------------------------------------------------------------------------- tool schemas
def _walk(schema, fn):
    fn(schema)
    for sub in (schema.get("properties") or {}).values():
        _walk(sub, fn)
    if isinstance(schema.get("items"), dict):
        _walk(schema["items"], fn)


def test_tool_schemas_are_strict_and_sorted():
    chief = chief_tool_definitions()
    names = [t["name"] for t in chief]
    assert names == sorted(names)
    assert set(names) == set(DATA_TOOL_NAMES) | {"consult_specialist", "create_specialist", "submit_decision"}
    spec = specialist_tool_definitions(["get_market_snapshot"])
    assert [t["name"] for t in spec] == ["get_market_snapshot", "submit_memo"]

    def check(s):
        if s.get("type") == "object":
            assert s["additionalProperties"] is False
        for bad in ("minimum", "maximum", "minLength", "maxLength", "minItems"):
            assert bad not in s
    for t in chief + spec:
        assert t["strict"] is True and t["description"]
        _walk(t["input_schema"], check)
    dec = next(t for t in chief if t["name"] == "submit_decision")["input_schema"]
    assert set(dec["required"]) == {"action", "scale", "forecast", "confidence", "horizon_bars", "rationale",
                                    "key_risks", "dissent"}
    assert dec["properties"]["action"]["enum"] == ["follow_quant", "scale", "veto", "override", "hold"]


def test_adhoc_whitelist_restricts_create_schema():
    tools = chief_tool_definitions(["get_calendar"])
    create = next(t for t in tools if t["name"] == "create_specialist")
    assert create["input_schema"]["properties"]["tools"]["items"]["enum"] == ["get_calendar"]


def test_schema_validator():
    schema = next(t for t in chief_tool_definitions() if t["name"] == "consult_specialist")["input_schema"]
    assert validate_against_schema(schema, {"role": "risk_officer", "question": "q"}) == []
    assert validate_against_schema(schema, {"role": "risk_officer"})
    assert validate_against_schema(schema, {"role": "cto", "question": "q"})
    assert validate_against_schema(schema, {"role": "risk_officer", "question": "q", "x": 1})
    assert validate_against_schema({"type": "number"}, True)  # bool is not a number


# ----------------------------------------------------------------------------- records
def _dec(**kw):
    base = {"action": "scale", "scale": 0.5, "forecast": 0.2, "confidence": 0.6, "horizon_bars": 12,
            "rationale": "r", "key_risks": ["a"], "dissent": ""}
    base.update(kw)
    return base


def test_decision_validation():
    d = Decision.from_tool_input(_dec())
    assert d.scale == 0.5 and d.key_risks == ("a",)
    assert Decision.from_tool_input(_dec(horizon_bars=12.0)).horizon_bars == 12
    for bad in (dict(scale=1.2), dict(forecast=-1.5), dict(confidence=True), dict(horizon_bars=0),
                dict(horizon_bars=2.5), dict(action="buy"), dict(rationale="  "),
                dict(key_risks=["x"] * 13), dict(scale=float("nan"))):
        with pytest.raises(RecordValidationError):
            Decision.from_tool_input(_dec(**bad))
    with pytest.raises(RecordValidationError) as exc:
        Decision.from_tool_input(_dec(scale=2, confidence=-1))
    assert len(exc.value.problems) == 2


def test_memo_validation():
    ok = {"stance": "bullish", "confidence": 0.6, "key_points": ["p"], "risks": [], "suggested_exposure": 0.4}
    m = Memo.from_tool_input(ok, agent_id="a", role="r", question="q")
    assert m.ok and m.for_chief()["stance"] == "bullish"
    for bad in ({"suggested_exposure": -0.2}, {"key_points": []}, {"stance": "long"}, {"confidence": 1.5}):
        with pytest.raises(RecordValidationError):
            Memo.from_tool_input({**ok, **bad}, agent_id="a", role="r", question="q")
    bearish = {**ok, "stance": "bearish", "suggested_exposure": 0.1}
    with pytest.raises(RecordValidationError):
        Memo.from_tool_input(bearish, agent_id="a", role="r", question="q")
    failed = Memo.failed(agent_id="a", role="r", question="q", status="refusal", error="declined")
    assert not failed.ok and failed.for_chief() == {"agent": "a", "role": "r", "status": "refusal",
                                                    "error": "declined"}


# ----------------------------------------------------------------------------- usage
def test_cost_estimate_with_cache_tokens_and_unknown_model():
    msg = message(text("x"), usage=FakeUsage(input_tokens=1000, output_tokens=100,
                                             cache_creation_input_tokens=2000, cache_read_input_tokens=10_000))
    u = estimate_response_cost(msg, requested_model="claude-opus-5")
    expected = (1000 * 5 + 2000 * 5 * 1.25 + 10_000 * 5 * 0.1 + 100 * 25) / 1e6
    assert u.cost_usd == pytest.approx(expected)
    assert u.total_tokens == 13_100
    u1h = estimate_response_cost(msg, requested_model="claude-opus-5", cache_ttl="1h")
    assert u1h.cost_usd == pytest.approx(expected + 2000 * 5 * 0.75 / 1e6)
    odd = message(text("x"), model="claude-unknown-9")
    worst = max(DEFAULT_PRICES.values(), key=lambda p: p.output_per_mtok)
    assert estimate_response_cost(odd, requested_model="claude-unknown-9").cost_usd == pytest.approx(
        (1000 * worst.input_per_mtok + 200 * worst.output_per_mtok) / 1e6)


def test_ledger_budget():
    ledger = UsageLedger(max_cost_usd=0.02, max_tokens=5000)
    assert ledger.exceeded() is None and ledger.fraction_used() == 0.0
    ledger.record("chief", message(text("x")), requested_model="claude-opus-5")
    assert ledger.fraction_used() == pytest.approx(max(0.01 / 0.02, 1200 / 5000))
    ledger.record("macro_strategist", message(text("x")), requested_model="claude-opus-5")
    assert "cost budget" in ledger.exceeded()
    s = ledger.summary()
    assert s["total"]["calls"] == 2 and set(s["by_agent"]) == {"chief", "macro_strategist"}


def test_custom_price_table_is_a_dict():
    prices = dict(DEFAULT_PRICES)
    from aurum.agents.config import ModelPrice
    prices["claude-opus-5"] = ModelPrice(1.0, 1.0)
    fake = FakeAnthropicClient({"chief": [message(decision_call())]})
    res = TradingDesk(StaticDeskDataProvider(), client=fake, config=DeskConfig(prices=prices),
                      journal_dir=None).run_cycle(NOW, 0.1)
    assert res.cost_usd == pytest.approx(1200 / 1e6)


# ----------------------------------------------------------------------------- misc
def test_sanitize_without_fallback_returns_same_objects():
    blocks = [thinking(), tool_use("get_positions")]
    out = sanitize_assistant_content(blocks)
    assert out == blocks and out[0] is blocks[0]


def test_journal_truncates_but_keeps_full_when_asked(tmp_path):
    j = CycleJournal(tmp_path / "j.jsonl", cycle_id="c1", max_chars=10)
    j.log("x", agent="a", payload="y" * 50)
    j.log("y", full=True, payload="z" * 50)
    lines = [json.loads(line) for line in (tmp_path / "j.jsonl").read_text().splitlines()]
    assert lines[0]["payload"].startswith("y" * 10) and "truncated 40 chars" in lines[0]["payload"]
    assert lines[1]["payload"] == "z" * 50
    assert lines[0]["agent"] == "a" and lines[0]["cycle_id"] == "c1"


def test_prompts_are_static_and_role_specific():
    for p in [CHIEF_SYSTEM_PROMPT, SPECIALIST_BASE_PROMPT, *ROLE_CHARTERS.values()]:
        assert not re.search(r"20\d\d-\d\d-\d\d", p)  # no dates -> cache-stable
    assert "capital preservation" in CHIEF_SYSTEM_PROMPT.lower()
    assert "dissent" in CHIEF_SYSTEM_PROMPT.lower()
    assert "not instructions" in SPECIALIST_BASE_PROMPT
    prompts = {r: specialist_system_prompt(r) for r in ROLE_CHARTERS}
    assert len(set(prompts.values())) == 4
    assert "Risk Officer" in prompts["risk_officer"]
    with pytest.raises(KeyError):
        specialist_system_prompt("astrologer")
    brief = chief_brief(as_of="t", quant_forecast=0.25, previous_forecast=None, mode="overlay",
                        max_abs_forecast=1.0, max_specialists=6, max_turns=8, max_cost_usd=3.0,
                        context={"account": "paper"})
    assert brief.startswith("Agent: chief") and "+0.2500" in brief and "never flip" in brief
    assert '{"account":"paper"}' in brief


def test_discretionary_brief_states_bounds():
    brief = chief_brief(as_of="t", quant_forecast=0.1, previous_forecast=0.2, mode="discretionary",
                        max_abs_forecast=0.5, max_specialists=6, max_turns=8, max_cost_usd=None)
    assert "[-0.50, +0.50]" in brief and "no cost cap" in brief


# ----------------------------------------------------------------------------- reviewer: adversarial
def test_declined_before_output_fallback_attempt_is_not_billed():
    """Docs: 'Declined-before-output attempts are reported but not billed'. Pricing the
    declined attempt's full prompt double-counts the (large) Chief context on every fallback."""
    msg = FakeMessage(
        content=[text("ok")], stop_reason="end_turn", model="claude-opus-4-8",
        usage=FakeUsage(input_tokens=40_000, output_tokens=500, iterations=[
            FakeIterationUsage("message", 40_000, 0, "claude-opus-5"),
            FakeIterationUsage("fallback_message", 40_000, 500, "claude-opus-4-8"),
        ]),
    )
    u = estimate_response_cost(msg, requested_model="claude-opus-5")
    assert u.cost_usd == pytest.approx((40_000 * 5 + 500 * 25) / 1e6)
    assert u.input_tokens == 40_000 and u.unbilled_input_tokens == 40_000
    # a mid-output decline (partial output generated) stays billed
    mid = FakeMessage(
        content=[text("ok")], stop_reason="end_turn", model="claude-opus-4-8",
        usage=FakeUsage(iterations=[
            FakeIterationUsage("message", 1_000, 300, "claude-opus-5"),
            FakeIterationUsage("fallback_message", 1_000, 200, "claude-opus-4-8"),
        ]),
    )
    assert estimate_response_cost(mid, requested_model="claude-opus-5").cost_usd == pytest.approx(
        (2_000 * 5 + 500 * 25) / 1e6)


def test_pre_output_refusal_is_not_billed():
    r = refusal(usage=FakeUsage(input_tokens=30_000, output_tokens=0))
    u = estimate_response_cost(r, requested_model="claude-opus-5")
    assert u.cost_usd == 0.0 and u.total_tokens == 0 and u.calls == 1
    # the whole chain declined before output (iterations form)
    chain = FakeMessage(content=[], stop_reason="refusal", usage=FakeUsage(iterations=[
        FakeIterationUsage("message", 30_000, 0, "claude-opus-5"),
        FakeIterationUsage("fallback_message", 30_000, 0, "claude-opus-4-8"),
    ]))
    assert estimate_response_cost(chain, requested_model="claude-opus-5").cost_usd == 0.0


def test_real_sdk_accepts_deadline_bounded_timeout(tmp_path):
    fake = FakeAnthropicClient({"chief": [message(decision_call(action="veto"))]})
    seen: list = []
    client = _mock_anthropic(fake, seen)
    cfg = DeskConfig(max_cycle_seconds=120.0)
    res = TradingDesk(StaticDeskDataProvider(), client=client, config=cfg, journal_dir=None).run_cycle(NOW, 0.4)
    assert fake.errors == [] and res.status == "decided" and res.final_forecast == 0.0
    headers, body = seen[0]
    assert "timeout" not in body  # a transport option, never serialised into the request body


# ----------------------------------------------------------------------------- reviewer 2: adversarial
def test_to_jsonable_missing_values_and_numpy_time_types_never_raise():
    import numpy as np

    from aurum.agents._json import dumps, to_jsonable
    obj = {
        "td": np.timedelta64(5, "m"), "td_nat": np.timedelta64("NaT", "s"), "dt": np.datetime64("2026-01-02T03:04"),
        "dt_nat": np.datetime64("NaT", "s"), "na": pd.NA, "nat": pd.NaT, "pytd": pd.Timedelta(minutes=2),
        "nullable": pd.array([1.5, None], dtype="Float64"), "idx": pd.DatetimeIndex(["2026-01-01"], tz="UTC"),
        "arr_dt": np.array(["2026-01-01", "NaT"], dtype="datetime64[s]"),
        "series": pd.Series([1.0, None], dtype="Float64"),
    }
    out = to_jsonable(obj)
    assert out["td"] == 300.0 and out["td_nat"] is None and out["dt"] == "2026-01-02T03:04:00"
    assert out["dt_nat"] is None and out["na"] is None and out["nat"] is None and out["pytd"] == 120.0
    assert out["nullable"] == [1.5, None] and out["idx"] == ["2026-01-01T00:00:00+00:00"]
    assert out["arr_dt"] == ["2026-01-01T00:00:00", None] and out["series"] == {"0": 1.0, "1": None}
    text = dumps(obj)
    assert "<NA>" not in text and "NaT" not in text

    class Weird:
        def __str__(self):
            raise RuntimeError("no")
    assert to_jsonable(Weird()) == "<unrepresentable>"


def test_validate_request_rejects_assistant_prefill():
    """Opus 4.6+ (incl. Opus 5) reject a request ending on an assistant turn with a 400; the
    fake must catch the desk producing one (it previously accepted it)."""
    base = [{"role": "user", "content": [{"type": "text", "text": "Agent: chief"}]}]
    prefill = base + [{"role": "assistant", "content": [{"type": "text", "text": "Working..."}]}]
    kw = build_request(cfg=AgentModelConfig(), system_prompt="S", tools=[], messages=prefill)
    assert any("prefill" in p for p in validate_request(kw))
    resume = base + [{"role": "assistant", "content": [
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}]}]
    kw = build_request(cfg=AgentModelConfig(), system_prompt="S", tools=[], messages=resume)
    assert validate_request(kw) == []


def test_ledger_budget_share():
    ledger = UsageLedger(max_cost_usd=0.10, max_tokens=None)
    ledger.record("chief", message(text("x"), usage=FakeUsage(input_tokens=10_000, output_tokens=1_000)),
                  requested_model="claude-opus-5")  # $0.075
    assert ledger.exceeded() is None
    assert "share of the cost budget" in ledger.exceeded(0.7)
    tok = UsageLedger(max_tokens=10_000)
    tok.record("a", message(text("x"), usage=FakeUsage(input_tokens=8_000, output_tokens=500)),
               requested_model="claude-opus-5")
    assert tok.exceeded() is None and "share of the token budget" in tok.exceeded(0.8)


def test_desk_config_rejects_bad_reserve_and_empty_prices():
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError):
            DeskConfig(chief_budget_reserve=bad)
    with pytest.raises(ValueError):
        DeskConfig(prices={})
    assert DeskConfig().specialist_budget_share == pytest.approx(0.8)
