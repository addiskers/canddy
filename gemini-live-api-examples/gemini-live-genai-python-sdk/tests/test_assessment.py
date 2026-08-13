"""assessment.py — rubric bands, hard validation, deterministic evidence checks,
eligibility gates, and the end-to-end scoring worker (the _generate network seam
is monkeypatched; no test ever talks to Gemini)."""

import asyncio
import copy
import uuid

import pytest

import assessment
import pricing
import store


# ── fixtures / builders ───────────────────────────────────────────────────────

def _transcript():
    return [
        {"role": "gemini", "text": "Please explain what happened on 6 August.",
         "ts": "2026-08-13T10:00:05+00:00"},
        {"role": "user", "text": "I stayed at my station and kept working the whole shift",
         "ts": "2026-08-13T10:00:20+00:00"},
        {"role": "gemini", "text": "Who first suggested stopping work?",
         "ts": "2026-08-13T10:01:00+00:00"},
        {"role": "user", "text": "I do not know who started it, I only heard shouting outside",
         "ts": "2026-08-13T10:01:20+00:00"},
        {"role": "gemini", "text": "Are you willing to follow policies going forward?",
         "ts": "2026-08-13T10:02:00+00:00"},
        {"role": "user", "text": "Yes, I will raise concerns through my supervisor next time",
         "ts": "2026-08-13T10:02:20+00:00"},
    ]


def _call(call_id=None, **over):
    c = {
        "id": call_id or ("as-" + uuid.uuid4().hex[:10]),
        "call_sid": "sid-assess", "source": "plivo", "caller": "+919824018000",
        "employee_id": "CNY-007", "campaign_id": None, "origin_call_id": None,
        "generation": 0,
        "started_at": "2026-08-13T10:00:00+00:00",
        "ended_at": "2026-08-13T10:10:00+00:00",
        "duration_seconds": 600, "language": "hi", "status": "completed",
        "booking_created": True, "gemini_model": "test-model",
        "tokens": pricing._empty_tokens(), "gemini_cost_usd": 0.0,
        "twilio": {"price_usd": None, "price_unit": None,
                   "status": None, "duration_seconds": None},
        "total_cost_usd": 0.0, "cost_estimated": False,
        "rsvp_outcome_status": "yes",
        "transcript": _transcript(), "tool_calls": [],
    }
    c.update(over)
    return c


_GOOD_SCORES = {"involvement": 26, "conduct": 18, "accountability": 18,
                "future_compliance": 14, "communication": 9, "overall_suitability": 4}


def _raw(scores=None, key_evidence=None, **over):
    raw = {
        "scores": dict(scores if scores is not None else _GOOD_SCORES),
        "classifications": {
            "incident_involvement": "Passive participant",
            "conduct": "Professional",
            "accountability": "Strong",
            "future_compliance": "Strong",
            "communication": "Clear",
            "red_flag_level": "None",
            "review_status": "Further consideration",
        },
        "key_evidence": key_evidence if key_evidence is not None else [
            {"quote": "I stayed at my station and kept working the whole shift",
             "turn": 2, "relevance": "describes passive conduct"}],
        "named_persons": [],
        "contradictions": [],
        "question_coverage": [{"q": 1, "status": "answered", "turn": 2},
                              {"q": 19, "status": "answered", "turn": 6}],
        "summary": "Employee describes staying at their station and commits to proper channels.",
    }
    raw.update(over)
    return raw


# ── band_for: every category edge (bands are code-owned, never the model's) ──

def test_band_boundaries_every_category_edge():
    bf = assessment.band_for
    # involvement (30): 25 / 15 / 5
    assert bf("involvement", 30) == "passive"
    assert bf("involvement", 25) == "passive"
    assert bf("involvement", 24) == "active_not_organising"
    assert bf("involvement", 15) == "active_not_organising"
    assert bf("involvement", 14) == "influenced_or_coordinated"
    assert bf("involvement", 5) == "influenced_or_coordinated"
    assert bf("involvement", 4) == "organiser_instigator"
    assert bf("involvement", 0) == "organiser_instigator"
    # conduct (20): 17 / 12 / 5
    assert bf("conduct", 17) == "respectful"
    assert bf("conduct", 16) == "participated_no_misconduct"
    assert bf("conduct", 12) == "participated_no_misconduct"
    assert bf("conduct", 11) == "improper_conduct"
    assert bf("conduct", 5) == "improper_conduct"
    assert bf("conduct", 4) == "abusive_threatening"
    # accountability (20): 17 / 12 / 5
    assert bf("accountability", 17) == "strong"
    assert bf("accountability", 16) == "partial"
    assert bf("accountability", 12) == "partial"
    assert bf("accountability", 11) == "limited"
    assert bf("accountability", 5) == "limited"
    assert bf("accountability", 4) == "none"
    # future_compliance (15): 13 / 9 / 4
    assert bf("future_compliance", 13) == "strong"
    assert bf("future_compliance", 12) == "accepts_with_reservations"
    assert bf("future_compliance", 9) == "accepts_with_reservations"
    assert bf("future_compliance", 8) == "conditional"
    assert bf("future_compliance", 4) == "conditional"
    assert bf("future_compliance", 3) == "may_repeat"
    # communication (10): 8 / 5
    assert bf("communication", 8) == "clear"
    assert bf("communication", 7) == "adequate"
    assert bf("communication", 5) == "adequate"
    assert bf("communication", 4) == "needs_review"
    # overall_suitability has no bands
    assert bf("overall_suitability", 5) == ""
    assert bf("overall_suitability", 0) == ""
    # out-of-range scores are clamped before banding
    assert bf("involvement", 99) == "passive"
    assert bf("involvement", -5) == "organiser_instigator"


# ── validate_result: clamped scores, recomputed total, checked enums ─────────

def test_validate_result_recomputes_total_from_clamped_scores():
    raw = _raw(scores={"involvement": 35, "conduct": -2, "accountability": 18,
                       "future_compliance": 14, "communication": 9,
                       "overall_suitability": 9})
    out = assessment.validate_result(raw)
    assert out["scores"]["involvement"] == {"score": 30, "max": 30, "band": "passive"}
    assert out["scores"]["conduct"] == {"score": 0, "max": 20, "band": "abusive_threatening"}
    assert out["scores"]["overall_suitability"] == {"score": 5, "max": 5}   # band "" omitted
    assert out["total_score"] == 30 + 0 + 18 + 14 + 9 + 5    # sum of CLAMPED scores
    assert out["classifications"]["red_flag_level"] == "None"


def test_validate_result_normalises_question_coverage():
    raw = _raw()
    raw["question_coverage"] = [
        {"q": 3, "status": "partial", "turn": 4},
        {"q": 1, "status": "answered", "turn": 2},
        {"q": 1, "status": "skipped"},              # duplicate → first wins
        {"q": 25, "status": "answered"},            # out of range → dropped
        {"q": 2, "status": "not-a-status"},         # bad enum → dropped
    ]
    out = assessment.validate_result(raw)
    assert out["question_coverage"] == [{"q": 1, "status": "answered", "turn": 2},
                                        {"q": 3, "status": "partial", "turn": 4}]


def test_validate_result_rejects_bad_shapes_and_enums():
    with pytest.raises(assessment.ValidationError):
        assessment.validate_result("not a dict")
    with pytest.raises(assessment.ValidationError):
        assessment.validate_result({})                       # missing scores
    raw = _raw()
    del raw["scores"]["conduct"]                             # missing category score
    with pytest.raises(assessment.ValidationError):
        assessment.validate_result(raw)
    raw = _raw()
    raw["classifications"]["red_flag_level"] = "High"        # not in RED_FLAG_LEVELS
    with pytest.raises(assessment.ValidationError):
        assessment.validate_result(raw)
    raw = _raw()
    raw["classifications"]["review_status"] = "Approved"     # not in REVIEW_STATUSES
    with pytest.raises(assessment.ValidationError):
        assessment.validate_result(raw)
    raw = _raw()
    del raw["classifications"]
    with pytest.raises(assessment.ValidationError):
        assessment.validate_result(raw)


# ── verify_evidence: quotes must exist verbatim in the transcript ────────────

def test_verify_evidence_verbatim_quote_verifies():
    out = assessment.verify_evidence(assessment.validate_result(_raw()), _call())
    assert out["key_evidence"][0]["verified"] is True
    assert out["evidence_verified"] is True
    assert out["evidence_check"] == {"checked": 1, "matched": 1, "unmatched": []}
    assert out["classifications"]["review_status"] == "Further consideration"  # not escalated


def test_verify_evidence_fabricated_quote_escalates_to_critical_review():
    fabricated = [{"quote": "I told everyone to stop working immediately",
                   "turn": 2, "relevance": "claims organising"}]
    out = assessment.verify_evidence(
        assessment.validate_result(_raw(key_evidence=fabricated)), _call())
    assert out["key_evidence"][0]["verified"] is False
    assert out["evidence_verified"] is False
    assert out["evidence_check"]["matched"] == 0
    assert out["classifications"]["review_status"] == "Critical human review"  # escalated


def test_organiser_band_with_zero_verified_quotes_is_rejected():
    """An 'instigator' score (involvement <= 4) with no verified supporting quote
    must never be emitted — the allegation firewall raises instead."""
    scores = dict(_GOOD_SCORES, involvement=2)
    result = assessment.validate_result(_raw(scores=scores, key_evidence=[]))
    with pytest.raises(assessment.ValidationError):
        assessment.verify_evidence(result, _call())
    # same with only a fabricated quote
    fabricated = [{"quote": "made-up admission of organising", "turn": 1, "relevance": "r"}]
    result = assessment.validate_result(_raw(scores=scores, key_evidence=fabricated))
    with pytest.raises(assessment.ValidationError):
        assessment.verify_evidence(result, _call())


# ── eligible(): the auto-scoring gate matrix ─────────────────────────────────

def test_eligible_matrix(monkeypatch):
    good = _call()
    monkeypatch.delenv("TT_ASSESSMENT_ENABLED", raising=False)      # default: disabled
    ok, reason = assessment.eligible(good)
    assert ok is False and "disabled" in reason

    monkeypatch.setenv("TT_ASSESSMENT_ENABLED", "true")
    assert assessment.eligible(good) == (True, "ok")                 # good plivo call
    assert assessment.eligible(_call(source="plivo_inbound"))[0] is True
    assert assessment.eligible(_call(source="browser"))[0] is False  # demo console excluded
    assert assessment.eligible(_call(duration_seconds=30))[0] is False        # short call
    assert assessment.eligible(_call(rsvp_outcome_status="voicemail"))[0] is False
    assert assessment.eligible(_call(rsvp_outcome_status="wrong_number"))[0] is False
    assert assessment.eligible(_call(status="in_progress"))[0] is False
    few_turns = _call(transcript=_transcript()[:4])                  # only 2 employee turns
    assert assessment.eligible(few_turns)[0] is False


# ── run_assessment: the worker end-to-end (fake _generate) ───────────────────

def _worker_env(monkeypatch):
    monkeypatch.setenv("TT_ASSESSMENT_ENABLED", "true")
    monkeypatch.setenv("TT_ASSESSMENT_MAX_ATTEMPTS", "1")            # no retry sleeps
    monkeypatch.setattr(assessment, "_sem", None)                    # asyncio primitives are loop-bound


def test_run_assessment_completes_and_promotes_flat_fields(monkeypatch):
    _worker_env(monkeypatch)
    call = _call()
    tokens = {"in": 10_000, "out": 1_000, "thoughts": 200}
    prompts = []

    async def fake_generate(prompt):
        prompts.append(prompt)
        return copy.deepcopy(_raw()), dict(tokens)

    monkeypatch.setattr(assessment, "_generate", fake_generate)

    async def run():
        await store.save_call(call)
        saved = await assessment.run_assessment(call["id"])
        listing = await store.list_calls({})
        meta = next(m for m in listing["items"] if m["id"] == call["id"])
        loaded = await store.load_call(call["id"])
        return saved, meta, loaded

    saved, meta, loaded = asyncio.run(run())
    assert saved is not None
    assert len(prompts) == 1                        # one clean pass, no retries
    a = loaded["assessment"]
    assert a["status"] == "completed"
    assert a["attempts"] == 1
    assert a["total_score"] == sum(_GOOD_SCORES.values())
    # human review is forced in CODE — the model schema/raw never carries the field
    assert "human_review_required" not in _raw()
    assert a["human_review_required"] is True
    assert a["tokens"] == tokens
    assert a["cost_usd"] == pricing.compute_assessment_cost(tokens)
    assert a["key_evidence"][0]["verified"] is True
    assert a["employee"]["phone"] == "+919824018000"
    # heavy-field invariant: the index meta carries FLAT fields, never the block
    assert "assessment" not in meta
    assert meta["assessment_status"] == "completed"
    assert meta["assessment_score"] == a["total_score"]
    assert meta["assessment_red_flag"] == "None"
    assert meta["assessment_review_status"] == "Further consideration"
    assert meta["assessment_involvement"] == "Passive participant"
    assert meta["assessment_reviewed"] is False
    # the scoring cost is folded into the call's running total
    assert loaded["total_cost_usd"] == pytest.approx(a["cost_usd"])


def test_run_assessment_failure_marks_failed_and_leaves_call_intact(monkeypatch):
    _worker_env(monkeypatch)
    call = _call()
    original_transcript = copy.deepcopy(call["transcript"])

    async def fake_generate(prompt):
        raise RuntimeError("model down")

    monkeypatch.setattr(assessment, "_generate", fake_generate)

    async def run():
        await store.save_call(call)
        result = await assessment.run_assessment(call["id"])
        return result, await store.load_call(call["id"])

    result, loaded = asyncio.run(run())
    assert result is None
    a = loaded["assessment"]
    assert a["status"] == "failed"
    assert a["attempts"] == 1                                        # attempts recorded
    assert "RuntimeError: model down" in a["error"]
    assert a["human_review_required"] is True
    # the rest of the call record is untouched
    assert loaded["status"] == "completed"
    assert loaded["rsvp_outcome_status"] == "yes"
    assert loaded["transcript"] == original_transcript
    # flat fields reflect the failure; no phantom score is promoted
    assert loaded["assessment_status"] == "failed"
    assert "assessment_score" not in loaded


def test_run_assessment_skips_ineligible_without_writing(monkeypatch):
    _worker_env(monkeypatch)
    call = _call(rsvp_outcome_status="voicemail")

    async def fake_generate(prompt):                                 # must never be reached
        raise AssertionError("generate called for an ineligible call")

    monkeypatch.setattr(assessment, "_generate", fake_generate)

    async def run():
        await store.save_call(call)
        result = await assessment.run_assessment(call["id"])
        return result, await store.load_call(call["id"])

    result, loaded = asyncio.run(run())
    assert result is None
    assert "assessment" not in loaded


# ── store.update_call: serialized load→mutate→save ───────────────────────────

def test_store_update_call_sequential_mutators_both_land():
    call = _call("upd-seq-1")

    async def run():
        await store.save_call(call)
        await store.update_call("upd-seq-1", lambda c: c.update(first="a"))
        await store.update_call("upd-seq-1", lambda c: c.update(second="b"))
        return await store.load_call("upd-seq-1")

    loaded = asyncio.run(run())
    assert loaded["first"] == "a"
    assert loaded["second"] == "b"                   # the second write kept the first
    assert loaded["rsvp_outcome_status"] == "yes"    # original fields intact


def test_store_update_call_missing_record_returns_none():
    async def run():
        return await store.update_call("upd-missing-xyz", lambda c: c.update(x=1))
    assert asyncio.run(run()) is None
