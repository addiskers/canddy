"""main.handle_record_interview — outcome normalization, incl. the voicemail coercions
that stop a machine answer from ever becoming a phantom 'employee requested callback',
plus the new interview-result fields (identity/refusal/language/progress)."""

from main import handle_record_interview


def _status(**kwargs):
    return handle_record_interview(**kwargs)["outcome_status"]


def test_valid_statuses_pass_through():
    for s in ("yes", "no", "callback", "voicemail", "do_not_contact", "wrong_number"):
        assert _status(outcome_status=s) == s


def test_unknown_status_with_voicemail_wording_coerces_to_voicemail():
    assert _status(outcome_status="voicemail_detected") == "voicemail"
    assert _status(outcome_status="voice mail") == "voicemail"
    assert _status(outcome_status="unknown", note="reached an answering machine") == "voicemail"


def test_unknown_status_with_refusal_flag_is_no():
    assert _status(outcome_status="refused", refused_interview=True) == "no"
    assert _status(outcome_status="", refused_interview=True) == "no"


def test_unknown_status_falls_back_to_callback_not_no():
    assert _status(outcome_status="not_interested") == "callback"
    assert _status(outcome_status="") == "callback"


def test_callback_with_voicemail_note_and_no_time_is_voicemail():
    # the exact old-prompt shape that created the phantom next-day-10am callbacks
    assert _status(outcome_status="callback",
                   note="voicemail — no live answer") == "voicemail"


def test_callback_with_voicemail_note_but_a_real_time_stays_callback():
    assert _status(outcome_status="callback", note="voicemail mentioned earlier",
                   callback_time_text="tomorrow evening") == "callback"
    assert _status(outcome_status="callback", note="voicemail",
                   callback_time_iso="2026-08-14T18:00:00+05:30") == "callback"


def test_genuine_employee_callback_untouched():
    assert _status(outcome_status="callback", note="busy at work") == "callback"
    assert _status(outcome_status="callback") == "callback"


# Interview result fields

def test_questions_completed_clamped_to_0_20():
    assert handle_record_interview(outcome_status="yes",
                                   questions_completed=20)["questions_completed"] == 20
    assert handle_record_interview(outcome_status="yes",
                                   questions_completed=25)["questions_completed"] == 20
    assert handle_record_interview(outcome_status="callback",
                                   questions_completed=-3)["questions_completed"] == 0
    assert handle_record_interview(outcome_status="callback",
                                   questions_completed="7")["questions_completed"] == 7
    assert handle_record_interview(outcome_status="callback",
                                   questions_completed="lots")["questions_completed"] == 0
    assert handle_record_interview(outcome_status="yes")["questions_completed"] == 0


def test_refused_interview_true_iff_status_no():
    # the flag follows the FINAL status, so a coerced outcome can never disagree with it
    assert handle_record_interview(outcome_status="no")["refused_interview"] is True
    assert handle_record_interview(outcome_status="unknown",
                                   refused_interview=True)["refused_interview"] is True
    for s in ("yes", "callback", "voicemail", "do_not_contact", "wrong_number"):
        r = handle_record_interview(outcome_status=s, refused_interview=True)
        assert r["refused_interview"] is False, s


def test_preferred_language_lowercased():
    assert handle_record_interview(outcome_status="yes",
                                   preferred_language="Hindi")["preferred_language"] == "hindi"
    assert handle_record_interview(outcome_status="yes",
                                   preferred_language=" GUJARATI ")["preferred_language"] == "gujarati"
    assert handle_record_interview(outcome_status="yes")["preferred_language"] == ""


def test_identity_flag_and_do_not_contact_derivation():
    r = handle_record_interview(outcome_status="yes", employee_confirmed_identity=True)
    assert r["employee_confirmed_identity"] is True
    assert r["do_not_contact"] is False
    r = handle_record_interview(outcome_status="do_not_contact")
    assert r["do_not_contact"] is True
