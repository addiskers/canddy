"""
Inbound call-back context — when a member dials the EO number back (usually after
seeing a missed call from us), look the caller up in the campaign contact list and
build the exact opening instruction ("trigger") the agent should follow, e.g.:

    "Hi Amman! Thanks for calling back — I tried reaching you a little while ago
     on behalf of EO Gujarat about the upcoming event, but I believe you were
     unavailable..."

All classification and date math happens HERE in Python — the model is never asked
to reason about timestamps or guess call history. Pure read-side helper: never
raises; an unknown number gets a safe generic-greeting trigger with NO invented
context (a wrong name or a made-up "we called you" is worse than a plain hello).
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import directory
import eo_db

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")

# Caller not in any campaign and not in the member directory.
UNKNOWN_TRIGGER = (
    "[INBOUND CALL: someone has just called the EO Gujarat number. You do NOT know "
    "who they are — never invent a name and never claim you called them. Greet them "
    "warmly, say you're speaking on behalf of EO Gujarat, and ask how you can help "
    "with the event. If they say we called them, apologise lightly for missing each "
    "other and continue with THE INVITATION.]"
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
    wins over the member directory; either may be empty."""
    name = str((cc or {}).get("name") or "").strip()
    if name:
        return name.split()[0]
    return directory.first_name_for(phone)


def _missed_trigger(first_name, when, agent_reason, member_line):
    """Trigger for the headline case: we tried them, failed, and they called back."""
    if first_name:
        return (
            f"[INBOUND CALL-BACK: {first_name} is calling US back — we tried calling them "
            f"{when} but {agent_reason}. Open with, in your own warm words keeping every fact: "
            f'"Hi {first_name}! Thanks for calling back — I tried reaching you {when} on behalf '
            f'of EO Gujarat about the upcoming event, {member_line}." Then continue straight '
            f"into THE INVITATION. Do NOT ask who they are, do NOT say \"am I speaking to…?\", "
            f"and mention the missed call only this once.]"
        )
    return (
        f"[INBOUND CALL-BACK: the caller is a member we tried calling {when} but "
        f"{agent_reason}. You were NOT given their name — never invent one. Open with, in "
        f'your own warm words: "Hello! Thanks for calling back — I tried reaching you {when} '
        f'on behalf of EO Gujarat about the upcoming event, {member_line}." Then continue '
        f"straight into THE INVITATION, and mention the missed call only this once.]"
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
        if first_name:  # in the member directory, but never part of a campaign
            out["trigger"] = (
                f"[INBOUND CALL: {first_name} is calling the EO Gujarat number. Greet them "
                f'warmly by name — "Hi {first_name}!" — say you\'re speaking on behalf of EO '
                f"Gujarat, and ask how you can help with the event. Continue with THE "
                f"INVITATION if they're interested. Do NOT claim we called them.]"
            )
        return out

    out["campaign_id"] = cc.get("campaign_id")
    outcome = str(cc.get("rsvp_outcome") or "").strip().lower()
    attempts = int(cc.get("attempts") or 0)
    when = _when_phrase(cc.get("last_attempt_at"), now=now)
    who = first_name or "the caller"

    if outcome == "voicemail":
        out["trigger"] = _missed_trigger(
            first_name, when,
            agent_reason="only reached their voicemail",
            member_line="but I'm afraid it went to your voicemail")
    elif outcome == "yes":
        out["trigger"] = (
            f"[INBOUND CALL: {who} is calling us and has ALREADY RSVP'd YES for the event. "
            f"Thank them warmly for calling, let them know they're on the list for the "
            f"evening, and ask how you can help. Do NOT re-invite and do NOT re-record — "
            f"only call record_rsvp again if they clearly change their answer.]"
        )
    elif outcome == "no":
        out["trigger"] = (
            f"[INBOUND CALL: {who} is calling us; they had earlier DECLINED the event "
            f"invitation. Thank them warmly for calling and ask how you can help. Don't "
            f"push the invitation — but if they say they've changed their mind and can now "
            f"join, delightedly record the new \"yes\".]"
        )
    elif outcome == "callback":
        out["trigger"] = (
            f"[INBOUND CALL-BACK: {who} is calling US back — they had earlier asked us to "
            f"call them back about the event. Thank them warmly for calling in ("
            f'"so glad you called!"), then continue straight into THE INVITATION. Do NOT '
            f"ask who they are.]"
        )
    elif outcome == "do_not_contact":
        out["trigger"] = (
            f"[INBOUND CALL: {who} is calling us. NOTE: they had asked not to be contacted "
            f"again — but THEY called US, so simply be warm and helpful and answer their "
            f"questions. Do NOT pitch the invitation unless they themselves ask to attend.]"
        )
    elif outcome == "wrong_number":
        out["trigger"] = UNKNOWN_TRIGGER   # number was confirmed not the member's
        out["name"] = ""
        out["campaign_id"] = None
    elif attempts > 0:
        out["trigger"] = _missed_trigger(
            first_name, when,
            agent_reason="could not reach them",
            member_line="but I believe you were unavailable")
    else:
        out["trigger"] = (
            f"[INBOUND CALL: {who} is calling the EO Gujarat number. They're on our list "
            f"for the event invitation but we had NOT called them yet — so do NOT claim we "
            f"tried calling. Greet warmly, say you're speaking on behalf of EO Gujarat, and "
            f"continue with THE INVITATION.]"
        )
    return out
