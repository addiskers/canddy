"""recorder.py — voicemail must never create an employee-callback; interview fields
persisted; mark_question progress accumulation; backprop exclusions."""

import asyncio
from datetime import datetime, timedelta, timezone

import eo_db
import recorder as recorder_mod
import store
from main import handle_mark_question, handle_record_interview
from recorder import CallRecorder


def _rec_with_call(**over):
    r = CallRecorder(model="test")
    r.call = {"id": "c1", "call_sid": "sid1", "caller": "+919000000001", "generation": 0,
              "campaign_id": None, "origin_call_id": None, "booking_created": False,
              "transcript": [], "tool_calls": []}
    r.call.update(over)
    return r


def _tool_event(result, name="record_interview"):
    return {"type": "tool_call", "name": name, "args": {}, "result": result}


def _interview_event(**kwargs):
    # result dicts exactly as main.handle_record_interview produces them
    return _tool_event(handle_record_interview(**kwargs))


def _mark_event(n, status="answered", gist=""):
    return _tool_event(handle_mark_question(question_number=n, status=status, gist=gist),
                       name="mark_question")


def test_voicemail_records_outcome_but_never_schedules_callback():
    r = _rec_with_call()
    r._record_tool(_interview_event(outcome_status="voicemail", note="machine answered"))
    assert r.call["rsvp_outcome_status"] == "voicemail"
    assert r.call["rsvp_note"] == "machine answered"
    assert "callback" not in r.call                      # no "employee requested callback"
    assert r.call["booking_created"] is False


def test_live_callback_still_schedules_a_callback_block():
    future = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()
    r = _rec_with_call()
    r._record_tool(_interview_event(outcome_status="callback",
                                    callback_time_iso=future,
                                    callback_time_text="this evening"))
    cb = r.call.get("callback")
    assert cb and cb["status"] == "pending"
    assert cb["to"] == "+919000000001"


def test_yes_after_pending_callback_cancels_it():
    r = _rec_with_call()
    r.call["callback"] = {"status": "pending"}
    r._record_tool(_interview_event(outcome_status="yes"))
    assert r.call["callback"]["status"] == "cancelled"
    assert r.call["booking_created"] is True             # "interview completed" flag


def test_interview_fields_and_note_are_persisted():
    r = _rec_with_call()
    r._record_tool(_interview_event(outcome_status="yes",
                                    employee_confirmed_identity=True,
                                    preferred_language="Hindi",
                                    questions_completed=20,
                                    note="cooperative throughout"))
    assert r.call["interview_confirmed_identity"] is True
    assert r.call["interview_refused"] is False
    assert r.call["interview_language"] == "hindi"
    assert r.call["interview_questions_completed"] == 20
    assert r.call["rsvp_note"] == "cooperative throughout"
    assert r.call["remark"] == "cooperative throughout"
    # the removed RSVP-era fields are never written any more
    assert "rsvp_guest_name" not in r.call
    assert "rsvp_accompanying_children" not in r.call


def test_refusal_is_persisted():
    r = _rec_with_call()
    r._record_tool(_interview_event(outcome_status="no", refused_interview=True))
    assert r.call["rsvp_outcome_status"] == "no"
    assert r.call["interview_refused"] is True
    assert r.call["booking_created"] is False


# mark_question: silent progress accumulation

def test_mark_question_accumulates_progress_and_never_touches_outcome():
    r = _rec_with_call()
    r._record_tool(_mark_event(1, "answered", "stayed at his station"))
    r._record_tool(_mark_event(2, "partial"))
    r._record_tool(_mark_event(3, "declined"))
    r._record_tool(_mark_event(2, "answered", "completed on re-ask"))   # update, not duplicate
    progress = r.call["interview_progress"]
    assert set(progress) == {"1", "2", "3"}
    assert progress["1"]["status"] == "answered"
    assert progress["1"]["gist"] == "stayed at his station"
    assert progress["2"]["status"] == "answered"                        # last write wins
    assert r.call["interview_questions_completed"] == 3
    assert "rsvp_outcome_status" not in r.call            # outcome untouched
    assert r.call["booking_created"] is False


def test_mark_question_ignores_out_of_range_numbers():
    r = _rec_with_call()
    r._record_tool(_mark_event(0))
    r._record_tool(_mark_event(21))
    assert "interview_progress" not in r.call


def test_record_interview_count_never_lowers_mark_question_progress():
    r = _rec_with_call()
    for n in (1, 2, 3, 4, 5):
        r._record_tool(_mark_event(n))
    assert r.call["interview_questions_completed"] == 5
    # the agent under-reports at record time — the accumulated count must survive
    r._record_tool(_interview_event(outcome_status="callback", questions_completed=2))
    assert r.call["interview_questions_completed"] == 5
    # ...but a HIGHER tool count may raise it
    r._record_tool(_interview_event(outcome_status="yes", questions_completed=20))
    assert r.call["interview_questions_completed"] == 20


def _run_backprop(monkeypatch, outcome):
    origin = {"id": "origin1", "callback": {"status": "pending", "result_call_id": None,
                                            "result_outcome": None}}
    saved, cc_calls = [], []

    async def fake_load(call_id):
        return origin

    async def fake_save(call):
        saved.append(call)

    monkeypatch.setattr(store, "load_call", fake_load)
    monkeypatch.setattr(store, "save_call", fake_save)
    monkeypatch.setattr(eo_db, "cc_set_outcome_by_phone",
                        lambda cid, phone, oc, **kw: cc_calls.append((cid, phone, oc)))

    r = _rec_with_call(origin_call_id="origin1", campaign_id=5,
                       rsvp_outcome_status=outcome)
    asyncio.run(r._backpropagate_to_origin())
    return origin, cc_calls


def test_backprop_voicemail_links_but_never_resolves_the_block(monkeypatch):
    origin, cc_calls = _run_backprop(monkeypatch, "voicemail")
    assert origin["callback"]["result_outcome"] == "voicemail"   # linked for history
    assert origin["callback"]["status"] == "pending"             # NOT resolved (not an answer)
    # the contact rollup DOES run — "Voicemail" is the truthful final label for a spent
    # callback chain (the old phantom was a forever-"Callback requested" contact)
    assert cc_calls == [(5, "+919000000001", "voicemail")]


def test_backprop_real_answer_resolves_block_and_rolls_onto_contact(monkeypatch):
    origin, cc_calls = _run_backprop(monkeypatch, "yes")
    assert origin["callback"]["status"] == "completed"
    assert cc_calls == [(5, "+919000000001", "yes")]
