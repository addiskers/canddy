"""
Inbound call-back context — when an employee dials the Canny AI screening number back
(usually after a missed call or our voicemail message), look the caller up in the
campaign contact list and build the exact opening instruction ("trigger") the
agent should follow.

All classification and date math happens HERE in Python — the model is never asked
to reason about timestamps or guess call history. Pure read-side helper: never
raises; an unknown number gets a safe generic-greeting trigger with NO invented
context.

Screening-specific hard rule baked into every trigger: this is a disciplinary
interview, and phones are shared — so even a roster-matched inbound number must
RE-CONFIRM the caller's identity by name before ANY incident content is spoken.
No trigger for an unconfirmed caller ever mentions the incident, the interview,
or the 6th of August.
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import directory
import eo_db
import store

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")

# Caller not in any campaign and not in the employee roster.
UNKNOWN_TRIGGER = (
    "[INBOUND CALL: someone has just called this number. You do NOT know who they "
    "are — never invent a name and never claim you called them. Greet politely in "
    "Hindi, say you're speaking on behalf of Canny management, and ask who is "
    "calling. Do NOT mention any incident, interview, or date to an unidentified "
    "caller. If they identify themselves as a Canny employee we tried to reach, "
    "confirm their name, then follow PURPOSE & CONSENT before any interview.]"
)

_IDENTITY_RULE = (
    "IDENTITY FIRST: even though this number matches {who}, phones are shared — "
    "CONFIRM by name that you are actually speaking with {who} before ANY mention "
    "of the incident or the interview. If it is someone else, say only that this "
    "is an official work matter from Canny management and ask when {who} can be "
    "reached."
)


def _when_phrase(last_attempt_iso, now=None):
    """Humanize an ISO timestamp into a spoken-style phrase, bucketed by IST calendar
    day: 'a little while ago' (<2h), 'earlier today', 'yesterday', 'a few days ago'."""
    try:
        dt = datetime.fromisoformat(str(last_attempt_iso))
    except (TypeError, ValueError):
        return "recently"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    delta = (now - dt).total_seconds()
    if delta < 0:
        return "recently"
    if delta < 2 * 3600:
        return "a little while ago"
    days = (now.astimezone(_IST).date() - dt.astimezone(_IST).date()).days
    if days <= 0:
        return "earlier today"
    if days == 1:
        return "yesterday"
    return "a few days ago"


def _first_name(cc, phone):
    """Campaign-contact name (the name we greeted them with on the outbound leg)
    wins over the employee roster; either may be empty."""
    name = str((cc or {}).get("name") or "").strip()
    if name:
        return name.split()[0]
    return directory.first_name_for(phone)


def _resume_hint(cc):
    """For an incomplete interview: which questions are already marked on the
    origin call, so the agent resumes at question N instead of restarting a
    15-minute interview. Sync read of one call JSON; never raises."""
    try:
        call_id = (cc or {}).get("last_call_id")
        if not call_id:
            return ""
        call = store._load_sync(call_id)
        if not call:
            return ""
        progress = call.get("interview_progress") or {}
        done = sorted(int(k) for k in progress.keys() if str(k).isdigit())
        if not done:
            return ""
        nxt = next((n for n in range(1, 11) if n not in done), 11)
        if nxt > 10:
            return ""
        return (f" Questions already covered last time: {', '.join(str(n) for n in done)}. "
                f"After consent, RESUME the interview from question {nxt} — do not re-ask "
                f"the covered ones (confirm briefly only if needed).")
    except Exception as e:
        logger.warning(f"resume hint lookup failed: {e}")
        return ""


def _missed_trigger(first_name, when, agent_reason):
    """Trigger for the headline case: we tried them, failed, and they called back."""
    if first_name:
        rule = _IDENTITY_RULE.format(who=first_name)
        return (
            f"[INBOUND CALL-BACK: this number belongs to {first_name}, a Canny contract "
            f"employee we tried calling {when} but {agent_reason}. {rule} Once {first_name} "
            f"is confirmed on the line: thank them for calling back, then give PURPOSE & "
            f"CONSENT and begin THE INTERVIEW. Mention the missed call only once.]"
        )
    return (
        f"[INBOUND CALL-BACK: the caller's number matches a Canny contract employee we "
        f"tried calling {when} but {agent_reason}. You were NOT given their name — never "
        f"invent one. Greet politely, say you're calling on behalf of Canny management, "
        f"and ask who is speaking. Only once they identify themselves as the employee we "
        f"tried to reach: thank them for calling back, then PURPOSE & CONSENT and THE "
        f"INTERVIEW. No incident content before that.]"
    )


def build(raw_from, now=None):
    """Context for an inbound call from `raw_from` (Plivo `From`, any format).

    Returns {"phone": E.164, "name": first name or "", "campaign_id": int|None,
             "trigger": opening instruction for the agent}. Never raises.
    """
    phone = directory.normalize_phone(raw_from)
    out = {"phone": phone, "name": "", "campaign_id": None, "trigger": UNKNOWN_TRIGGER}
    if not phone:
        return out

    try:
        cc = eo_db.cc_find_recent_by_phone(phone)
    except Exception as e:
        logger.warning(f"Inbound context lookup failed for {phone}: {e}")
        cc = None

    first_name = _first_name(cc, phone)
    out["name"] = first_name

    if not cc:
        if first_name:  # in the employee roster, but never part of a screening campaign
            rule = _IDENTITY_RULE.format(who=first_name)
            out["trigger"] = (
                f"[INBOUND CALL: this number matches {first_name} on the Canny employee "
                f"roster, but no screening call is on record for them. {rule} Once "
                f"confirmed, say you're speaking on behalf of Canny management and ask "
                f"how you can help. Do NOT claim we called them, and do NOT start an "
                f"interview that isn't scheduled for them.]"
            )
        return out

    out["campaign_id"] = cc.get("campaign_id")
    outcome = str(cc.get("rsvp_outcome") or "").strip().lower()
    attempts = int(cc.get("attempts") or 0)
    when = _when_phrase(cc.get("last_attempt_at"), now=now)
    who = first_name or "the employee"
    rule = _IDENTITY_RULE.format(who=who)

    if outcome == "voicemail":
        out["trigger"] = (
            f"[INBOUND CALL-BACK: this number belongs to {who}, a Canny contract employee. "
            f"We tried calling {when} and left a brief voicemail asking them to call back. "
            f"{rule} Once confirmed: thank them for returning the call, then give PURPOSE & "
            f"CONSENT and begin THE INTERVIEW.]"
        )
    elif outcome == "yes":
        out["trigger"] = (
            f"[INBOUND CALL: this number belongs to {who}, whose screening interview is "
            f"ALREADY COMPLETED. {rule} Once confirmed: thank them for calling, tell them "
            f"their interview is already on record and nothing more is needed from them, "
            f"answer brief practical questions per WHAT YOU KNOW, and close politely. "
            f"Do NOT redo the interview and do NOT re-record — only call record_interview "
            f"again if something genuinely new must be noted.]"
        )
    elif outcome == "no":
        out["trigger"] = (
            f"[INBOUND CALL: this number belongs to {who}, who had earlier DECLINED to "
            f"participate in the screening interview. {rule} Once confirmed: thank them "
            f"for calling and ask how you can help. Do NOT pressure them — but if they "
            f"say they now wish to give their side, give PURPOSE & CONSENT and run the "
            f"full INTERVIEW, and record the new outcome.]"
        )
    elif outcome == "callback":
        resume = _resume_hint(cc)
        out["trigger"] = (
            f"[INBOUND CALL-BACK: this number belongs to {who}, who had asked us to call "
            f"back about the screening interview (or the last interview was left "
            f"incomplete). {rule} Once confirmed: thank them for calling in, then give "
            f"PURPOSE & CONSENT and continue THE INTERVIEW.{resume}]"
        )
    elif outcome == "do_not_contact":
        out["trigger"] = (
            f"[INBOUND CALL: this number belongs to {who}. NOTE: they had asked not to be "
            f"contacted again — but THEY called US, so simply be polite and helpful and "
            f"answer their questions per WHAT YOU KNOW. {rule} Do NOT start the interview "
            f"unless they themselves clearly ask to give their statement now.]"
        )
    elif outcome == "wrong_number":
        out["trigger"] = UNKNOWN_TRIGGER   # number was confirmed not the employee's
        out["name"] = ""
        out["campaign_id"] = None
    elif attempts > 0:
        out["trigger"] = _missed_trigger(
            first_name, when,
            agent_reason="could not reach them")
    else:
        out["trigger"] = (
            f"[INBOUND CALL: this number belongs to {who}, who is on the screening list "
            f"but we had NOT called them yet — so do NOT claim we tried calling. {rule} "
            f"Once confirmed: say you're calling on behalf of Canny management about an "
            f"official matter, give PURPOSE & CONSENT, and begin THE INTERVIEW.]"
        )
    return out
