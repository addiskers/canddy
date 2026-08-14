"""Canny AI — post-call interview assessment.

QUESTIONS is the single source of truth for the 10-question Canny/Baoxhin
screening: gemini_live.py renders it into the live interviewer's system prompt,
and the scorer below renders it into the assessment prompt, so the two can
never drift.

The scoring pipeline (added further down) runs a Gemini text pass over a
completed call's transcript and writes an `assessment` block onto the call
record. It never blocks or alters the rest of the call record, and every
result carries human_review_required=True — the AI never decides employment
outcomes.
"""

RUBRIC_VERSION = 1

# The 10 core interview questions (client-reduced set, Aug 2026: merged incident/
# return-to-work items, added apology-letter and continue-working questions).
# en = canonical English text (used by the scorer), hi = phrasing hint for the
# live agent's Hindi-first delivery, followups = asked only when needed.
QUESTIONS = [
    {
        "n": 1, "key": "incident_account",
        "en": "Please explain, in your own words, what happened on 6 August from the time you first became aware of the phone-storage requirement until the work stoppage.",
        "hi": "6 अगस्त को फ़ोन जमा करने के नियम का पता चलने से लेकर काम रुकने तक जो कुछ हुआ, वह अपने शब्दों में बताइए।",
        "followups": "What happened after that? | What did you personally do?",
    },
    {
        "n": 2, "key": "learned_of_policy",
        "en": "How did you first come to know about the phone-storage requirement, and what were you told about it?",
        "hi": "फ़ोन जमा करने के नियम के बारे में आपको सबसे पहले कैसे पता चला, और आपको क्या बताया गया था?",
        "followups": "Who told you about it?",
    },
    {
        "n": 3, "key": "personal_concern",
        "en": "What was your personal concern with the phone-storage requirement?",
        "hi": "इस नियम को लेकर आपकी अपनी क्या चिंता थी?",
        "followups": "Was your concern about the policy itself or how it was being implemented?",
    },
    {
        "n": 4, "key": "raised_before_stopping",
        "en": "Before you stopped working, did you raise your concern with your Canny supervisor, Canny management, or Baoxhin management?",
        "hi": "काम रोकने से पहले, क्या आपने अपनी चिंता Canny सुपरवाइज़र, Canny management या Baoxhin management को बताई थी?",
        "followups": "If no: Why did you not raise it with them? Kiske paas complaint raise ki thi, strike karne se pehle?",
    },
    {
        "n": 5, "key": "own_decision",
        "en": "Was stopping work your own decision, or did someone ask, encourage, or pressure you to participate?",
        "hi": "काम रोकना आपका अपना फ़ैसला था, या किसी ने आपसे कहा, आपको समझाया या आप पर दबाव डाला?",
        "followups": "If someone: Kisne bola tha? | Exactly kya bola tha? | Aapne uske baad khud kya kiya?",
    },
    {
        "n": 6, "key": "who_initiated",
        "en": "Who first suggested that employees should stop working, and what was said?",
        "hi": "सबसे पहले किसने कहा कि सबको काम रोक देना चाहिए, और क्या कहा गया था?",
        "followups": "Where did this discussion take place? | Did you personally ask any other employee to stop working?",
    },
    {
        "n": 7, "key": "return_to_work",
        "en": "Were you asked to return to work? If yes, what did you do? Did anyone discourage, pressure, or prevent you from returning?",
        "hi": "क्या आपसे वापस काम पर लौटने को कहा गया था? अगर हाँ, तो आपने क्या किया? क्या किसी ने आपको लौटने से रोका या दबाव डाला?",
        "followups": "Did YOU discourage or prevent anyone else from returning?",
    },
    {
        "n": 8, "key": "own_conduct",
        "en": "During the incident, did you personally use any abusive, threatening, or inappropriate language or behaviour towards any Canny or Baoxhin representative?",
        "hi": "उस दौरान, क्या आपने ख़ुद Canny या Baoxhin के किसी व्यक्ति के साथ गाली-गलौज, धमकी या ग़लत भाषा या व्यवहार किया?",
        "followups": "If yes: Please explain what happened.",
    },
    {
        "n": 9, "key": "apology_letter",
        "en": "Were you asked to sign an apology letter regarding the incident? If yes, did you sign it?",
        "hi": "क्या आपसे incident के बारे में apology letter sign करने को कहा गया था? अगर हाँ, तो क्या आपने sign किया?",
        "followups": "Did you understand what you were signing? | Did you sign it voluntarily? | Did you have any concerns about signing it?",
    },
    {
        "n": 10, "key": "continue_working",
        "en": "Do you want to continue working at Baoxhin through Canny, and are you ready to follow the applicable Baoxhin policies and management instructions?",
        "hi": "अब एक simple question का सीधा जवाब दीजिए — क्या आप Canny के साथ अपनी नौकरी continue करना चाहते हैं, और Baoxhin site पर applicable policies और management instructions follow करने के लिए ready हैं?",
        "followups": "If YES: confirm — future mein concern ho toh kaam rokne ke bajay Canny management ke proper channel par raise karna hoga; is baat ko aap clearly samajh rahe hain? | If NO: Kya aap apni position clearly confirm karna chahenge ki aap Canny ke saath employment continue nahi karna chahte? | If UNSURE: Aapko kya concern hai jo aapko decision lene se rok raha hai?",
    },
]

QUESTIONS_TOTAL = len(QUESTIONS)


def questions_prompt_block():
    """Render QUESTIONS as the numbered list embedded in the live agent's prompt."""
    lines = []
    for q in QUESTIONS:
        lines.append(f"{q['n']}. {q['en']}")
        lines.append(f"   हिंदी: \"{q['hi']}\"")
        if q["followups"]:
            lines.append(f"   Follow-up ONLY if the answer needs it: {q['followups']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone

import pricing
import store

logger = logging.getLogger(__name__)

_client_instance = None
_tasks = set()            # strong refs — bare create_task results are GC-able
_sem = None               # lazy: created on the running loop


# Category rubric: (max points, [(min_score_inclusive, band), ...] highest first).
# Bands are ALWAYS recomputed in code from the numeric score — the model's own
# band label is never trusted.
CATEGORIES = {
    "involvement":         (30, [(25, "passive"), (15, "active_not_organising"),
                                 (5, "influenced_or_coordinated"), (0, "organiser_instigator")]),
    "conduct":             (20, [(17, "respectful"), (12, "participated_no_misconduct"),
                                 (5, "improper_conduct"), (0, "abusive_threatening")]),
    "accountability":      (20, [(17, "strong"), (12, "partial"), (5, "limited"), (0, "none")]),
    "future_compliance":   (15, [(13, "strong"), (9, "accepts_with_reservations"),
                                 (4, "conditional"), (0, "may_repeat")]),
    "communication":       (10, [(8, "clear"), (5, "adequate"), (0, "needs_review")]),
    "overall_suitability": (5,  [(0, "")]),
}

INVOLVEMENT_LEVELS = ["Passive participant", "Active participant", "Influenced others",
                      "Coordinator or organiser", "Potential instigator", "Unclear"]
CONDUCT_LEVELS = ["Professional", "Concerning", "Abusive or aggressive", "Threatening",
                  "Other serious concern"]
ACCOUNTABILITY_LEVELS = ["Strong", "Partial", "Limited", "None"]
COMPLIANCE_LEVELS = ["Strong", "Moderate", "Unclear", "Poor"]
COMMUNICATION_LEVELS = ["Clear", "Adequate", "Needs review"]
RED_FLAG_LEVELS = ["None", "Moderate", "Critical"]
REVIEW_STATUSES = ["Pass", "Moderate", "Needs review"]

# Records scored before the 3-status simplification carry the old four values;
# reads normalise them so no data migration is needed.
LEGACY_REVIEW_MAP = {
    "Further consideration": "Pass",
    "Canny review": "Moderate",
    "Baoxhin review": "Moderate",
    "Critical human review": "Needs review",
}


def normalize_review_status(value):
    """Map a legacy review status to the current 3-value set; passthrough otherwise."""
    return LEGACY_REVIEW_MAP.get(value, value)
COVERAGE_STATUSES = ["answered", "partial", "skipped", "refused"]

# Gemini structured-output schema (OpenAPI subset). human_review_required is
# deliberately ABSENT — code forces it True on every write.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {k: {"type": "integer"} for k in CATEGORIES},
            "required": list(CATEGORIES),
        },
        "classifications": {
            "type": "object",
            "properties": {
                "incident_involvement": {"type": "string", "enum": INVOLVEMENT_LEVELS},
                "conduct": {"type": "string", "enum": CONDUCT_LEVELS},
                "accountability": {"type": "string", "enum": ACCOUNTABILITY_LEVELS},
                "future_compliance": {"type": "string", "enum": COMPLIANCE_LEVELS},
                "communication": {"type": "string", "enum": COMMUNICATION_LEVELS},
                "red_flag_level": {"type": "string", "enum": RED_FLAG_LEVELS},
                "review_status": {"type": "string", "enum": REVIEW_STATUSES},
            },
            "required": ["incident_involvement", "conduct", "accountability",
                         "future_compliance", "communication", "red_flag_level",
                         "review_status"],
        },
        "key_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "turn": {"type": "integer"},
                    "translation": {"type": "string"},
                    "relevance": {"type": "string"},
                },
                "required": ["quote", "turn", "relevance"],
            },
        },
        "named_persons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "alleged_conduct": {"type": "string"},
                    "turn": {"type": "integer"},
                },
                "required": ["name", "alleged_conduct"],
            },
        },
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "turns": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["description"],
            },
        },
        "question_coverage": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "q": {"type": "integer"},
                    "status": {"type": "string", "enum": COVERAGE_STATUSES},
                    "turn": {"type": "integer"},
                },
                "required": ["q", "status"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["scores", "classifications", "key_evidence", "named_persons",
                 "question_coverage", "summary"],
}


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _enabled():
    return (os.getenv("TT_ASSESSMENT_ENABLED", "false").strip().lower()
            in ("1", "true", "yes", "on"))


def _model_name():
    return os.getenv("TT_ASSESSMENT_MODEL", "gemini-3.6-flash").strip() or "gemini-3.6-flash"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def band_for(category, score):
    """The rubric band for a category score (code-owned; the model's label never wins)."""
    max_pts, bands = CATEGORIES[category]
    score = max(0, min(max_pts, int(score)))
    for floor, band in bands:
        if score >= floor:
            return band
    return bands[-1][1]


def _clamp(category, score):
    max_pts, _ = CATEGORIES[category]
    try:
        return max(0, min(max_pts, int(score)))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Eligibility & scheduling
# ---------------------------------------------------------------------------

def eligible(call):
    """(bool, reason) — should this call be auto-scored? Pure function."""
    if not _enabled():
        return False, "assessment disabled (TT_ASSESSMENT_ENABLED)"
    if (call or {}).get("status") != "completed":
        return False, f"call status is {(call or {}).get('status')}"
    source = call.get("source") or ""
    browser_ok = (os.getenv("TT_ASSESS_BROWSER", "false").strip().lower()
                  in ("1", "true", "yes", "on"))
    if source not in ("plivo", "plivo_inbound") and not (browser_ok and source == "browser"):
        return False, f"source {source} not assessed"
    min_secs = _env_int("TT_ASSESSMENT_MIN_SECONDS", 90)
    if (call.get("duration_seconds") or 0) < min_secs:
        return False, f"shorter than {min_secs}s"
    outcome = call.get("rsvp_outcome_status") or ""
    if outcome in ("voicemail", "wrong_number", "do_not_contact"):
        return False, f"outcome {outcome}"
    employee_turns = sum(1 for t in (call.get("transcript") or [])
                         if t.get("role") == "user")
    if employee_turns < 3:
        return False, "fewer than 3 employee turns"
    return True, "ok"


def maybe_schedule(call, trigger="auto", requested_by="system"):
    """Fire-and-forget scheduling from recorder.close(). Sync; never raises."""
    try:
        ok, reason = eligible(call)
        if not ok:
            logger.info(f"assessment skipped for {(call or {}).get('id')}: {reason}")
            return False
        schedule(call["id"], trigger=trigger, requested_by=requested_by)
        return True
    except Exception as e:
        logger.warning(f"assessment.maybe_schedule failed: {e}")
        return False


def schedule(call_id, trigger="manual", requested_by="system", force=False):
    """Create the scoring task on the running loop (single-worker pattern)."""
    t = asyncio.create_task(run_assessment(call_id, trigger=trigger,
                                           requested_by=requested_by, force=force))
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)
    return t


def _get_sem():
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(_env_int("TT_ASSESSMENT_MAX_CONCURRENT", 2))
    return _sem


# ---------------------------------------------------------------------------
# Prompt & transcript rendering
# ---------------------------------------------------------------------------

def render_transcript(call):
    """Numbered turns the evidence quotes must cite: [#07 02:31 EMPLOYEE] ..."""
    started = None
    try:
        started = datetime.fromisoformat(str(call.get("started_at")))
    except (TypeError, ValueError):
        pass
    lines = []
    for i, t in enumerate(call.get("transcript") or [], start=1):
        role = "INTERVIEWER" if t.get("role") == "gemini" else "EMPLOYEE"
        offset = "--:--"
        try:
            ts = datetime.fromisoformat(str(t.get("ts")))
            if started is not None:
                secs = max(0, int((ts - started).total_seconds()))
                offset = f"{secs // 60:02d}:{secs % 60:02d}"
        except (TypeError, ValueError):
            pass
        lines.append(f"[#{i:02d} {offset} {role}] {t.get('text') or ''}")
    return "\n".join(lines)


def _questions_scoring_block():
    return "\n".join(f"{q['n']}. {q['en']}" for q in QUESTIONS)


def build_prompt(call, validator_error=""):
    duration = call.get("duration_seconds") or 0
    language = call.get("language") or call.get("interview_language") or "unknown"
    err = ""
    if validator_error:
        err = (f"\n\nIMPORTANT — your previous output failed validation: {validator_error}. "
               f"Correct this in your new output.")
    return f"""You are a neutral HR assessment analyst for Canny Management Services. You are reviewing ONE employee's screening-interview transcript about the 6 August 2026 collective work stoppage at the Baoxhin facility. You score strictly by the rubric below. You never decide employment outcomes — every assessment goes to human management review.

## NON-NEGOTIABLE FAIRNESS RULES (apply before anything else)
- Judge the SUBSTANCE of the answers and the conduct the employee themselves describes — never how they sound.
- Answers in Gujarati or Hindi are fully valid evidence. NEVER adjust any score for language choice, accent, fluency, grammar, vocabulary, or limited English.
- Tone is a secondary contextual signal only. A polite tone never reduces the seriousness of an admission (e.g. having instructed others to stop work); a blunt or emotional tone is not misconduct.
- Interview language for this call: {language} (context only — never a penalty).

## SCORING RUBRIC — 100 points
A. involvement (30): 25-30 passive participant/follower, no evidence of influencing others; 15-24 active participant, no clear evidence of organising or instigating; 5-14 influenced, encouraged or coordinated other employees; 0-4 strong evidence of organising, instigating, directing or actively escalating the stoppage. Never classify someone an organiser solely because they were named — only their OWN statements in THIS transcript count for their score.
B. conduct (20): 17-20 respectful conduct, no serious misconduct; 12-16 participated in the stoppage but no serious misconduct; 5-11 inappropriate language, pressure on others, refusal of reasonable instructions; 0-4 threatening, abusive, intimidating, aggressive or seriously disruptive conduct.
C. accountability (20): 17-20 clearly acknowledges actions, recognises what could have been handled differently; 12-16 accepts some responsibility but significant justification/blame; 5-11 minimises or repeatedly denies responsibility; 0-4 no accountability, fully justifies the conduct, or indicates willingness to repeat it.
D. future_compliance (15): 13-15 clearly willing to use proper escalation channels and keep working while concerns are addressed; 9-12 generally accepts with reservations; 4-8 unclear or conditional; 0-3 indicates they may again join a collective refusal/stoppage.
E. communication (10): can they explain their position, answer directly, understand escalation, disagree professionally — in WHATEVER language. This is NOT an English-fluency test.
F. overall_suitability (5): general suitability for a structured workplace.

## RED-FLAG LEVELS
- Critical: credible evidence (from their OWN statements) of organising/instigating; directing others to stop; actively pressuring others; preventing or discouraging return to work; threats or intimidation; serious abusive conduct; unauthorised promises about jobs/salary; explicit indication the conduct would be repeated.
- Moderate: participation with limited accountability; significant blame-shifting; evasive or contradictory answers; uncertain future compliance; admitted influence over others where the extent is unclear.
- None: no evidence of organising; clear distinction between their own and others' actions; accountability; willingness to comply.
Set review_status accordingly: "Needs review" for Critical; "Moderate" for Moderate concerns; "Pass" otherwise.

## EVIDENCE RULES (hard constraints)
- Every serious finding MUST cite what THIS employee actually said: a VERBATIM quote in the original language, with the turn number from the transcript below. Do not paraphrase inside "quote" — copy the exact words.
- Statements about OTHER employees are allegations, not facts: put them ONLY in named_persons (name + what conduct this employee described + turn). They never change this employee's scores. A name alone is not organiser evidence.
- Contradictory or uncertain answers: list them in contradictions for human review — never resolve them by assumption.
- If the transcript does not support a finding, do not make it.
- Next to any non-English quote, put an English translation in "translation".

## THE {QUESTIONS_TOTAL} CORE QUESTIONS (emit question_coverage: one entry per question that was raised, with status answered/partial/skipped/refused and the turn where it was answered)
(Q9, the apology letter, is context for accountability; Q10, continuation willingness, feeds future_compliance. Grievances the employee raised but the interviewer only noted — food, salary, accommodation, supervisor behaviour — are context, never conduct findings by themselves.)
{_questions_scoring_block()}

## INTERVIEW TRANSCRIPT
Interview metadata: phone {call.get('caller') or 'unknown'}, employee_id {call.get('employee_id') or 'unknown'}, duration {duration // 60}:{duration % 60:02d}, language {language}, date {str(call.get('started_at') or '')[:10]}.
{render_transcript(call)}

## OUTPUT
Fill the JSON schema exactly. "summary" is a 3-5 sentence factual narrative for a manager: what this employee says they did, their accountability, and their stated future intent — no speculation, no verdicts. Scores must be integers within each category's maximum.{err}"""


# ---------------------------------------------------------------------------
# Model call (seam for tests) & validation
# ---------------------------------------------------------------------------

def _client():
    global _client_instance
    if _client_instance is None:
        from google import genai
        _client_instance = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client_instance


async def _generate(prompt):
    """One structured-output scoring call. Returns (raw_dict, tokens_dict).
    Monkeypatched in tests — keep the signature stable."""
    from google.genai import types
    cfg_kwargs = {
        "temperature": 0,
        "response_mime_type": "application/json",
        "response_schema": RESPONSE_SCHEMA,
    }
    if hasattr(types, "ThinkingConfig"):
        try:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=1024)
        except Exception:
            pass
    resp = await _client().aio.models.generate_content(
        model=_model_name(),
        contents=prompt,
        config=types.GenerateContentConfig(**cfg_kwargs),
    )
    text = resp.text or ""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    raw = json.loads(text)
    um = getattr(resp, "usage_metadata", None)
    tokens = {
        "in": getattr(um, "prompt_token_count", 0) or 0,
        "out": getattr(um, "candidates_token_count", 0) or 0,
        "thoughts": getattr(um, "thoughts_token_count", 0) or 0,
    }
    return raw, tokens


class ValidationError(Exception):
    pass


def validate_result(raw):
    """Hard validation + normalisation. Never trusts the model: scores clamped,
    bands recomputed from scores, total recomputed from the sum, enums checked.
    Raises ValidationError on unusable output."""
    if not isinstance(raw, dict):
        raise ValidationError("output is not a JSON object")
    raw_scores = raw.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValidationError("missing scores object")
    scores = {}
    total = 0
    for cat, (max_pts, _) in CATEGORIES.items():
        if cat not in raw_scores:
            raise ValidationError(f"missing score for {cat}")
        s = _clamp(cat, raw_scores.get(cat))
        entry = {"score": s, "max": max_pts}
        band = band_for(cat, s)
        if band:
            entry["band"] = band
        scores[cat] = entry
        total += s

    cls = raw.get("classifications")
    if not isinstance(cls, dict):
        raise ValidationError("missing classifications object")
    checked = {}
    for field, allowed in (
            ("incident_involvement", INVOLVEMENT_LEVELS), ("conduct", CONDUCT_LEVELS),
            ("accountability", ACCOUNTABILITY_LEVELS), ("future_compliance", COMPLIANCE_LEVELS),
            ("communication", COMMUNICATION_LEVELS), ("red_flag_level", RED_FLAG_LEVELS),
            ("review_status", REVIEW_STATUSES)):
        v = cls.get(field)
        if v not in allowed:
            raise ValidationError(f"classifications.{field}={v!r} not in {allowed}")
        checked[field] = v

    def _items(key, required):
        out = []
        for item in raw.get(key) or []:
            if isinstance(item, dict) and all(item.get(f) is not None for f in required):
                out.append(item)
        return out

    coverage = []
    seen_q = set()
    for item in _items("question_coverage", ("q", "status")):
        try:
            qn = int(item["q"])
        except (TypeError, ValueError):
            continue
        if 1 <= qn <= QUESTIONS_TOTAL and qn not in seen_q and item["status"] in COVERAGE_STATUSES:
            seen_q.add(qn)
            coverage.append({"q": qn, "status": item["status"],
                             "turn": item.get("turn")})
    coverage.sort(key=lambda x: x["q"])

    return {
        "total_score": total,
        "scores": scores,
        "classifications": checked,
        "key_evidence": [
            {"quote": str(i["quote"]), "turn": int(i["turn"]),
             "translation": str(i.get("translation") or ""),
             "relevance": str(i["relevance"])}
            for i in _items("key_evidence", ("quote", "turn", "relevance"))],
        "named_persons": [
            {"name": str(i["name"]), "alleged_conduct": str(i["alleged_conduct"]),
             "turn": i.get("turn"),
             "note": "allegation only — not verified fact"}
            for i in _items("named_persons", ("name", "alleged_conduct"))],
        "contradictions": [
            {"description": str(i["description"]), "turns": i.get("turns") or []}
            for i in _items("contradictions", ("description",))],
        "question_coverage": coverage,
        "summary": str(raw.get("summary") or "").strip(),
    }


def _norm_text(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]", " ", str(s or "").lower())).strip()


def verify_evidence(result, call):
    """Deterministic quote check — no second LLM pass. A quote that appears
    nowhere in the transcript is treated as fabricated: the item is marked
    unverified and review_status is ESCALATED (never lowered; scores untouched)."""
    transcript = call.get("transcript") or []
    norm_turns = [_norm_text(t.get("text")) for t in transcript]
    all_text = " ".join(norm_turns)
    checked = matched = 0
    unmatched = []
    for item in result.get("key_evidence") or []:
        checked += 1
        q = _norm_text(item.get("quote"))
        turn = item.get("turn") or 0
        ok = False
        if q:
            lo, hi = max(1, turn - 1), min(len(norm_turns), turn + 1)
            ok = any(q in norm_turns[i - 1] for i in range(lo, hi + 1)) if norm_turns else False
            if not ok and q in all_text:
                ok = True                     # right words, wrong turn number — not fabricated
        item["verified"] = ok
        if ok:
            matched += 1
        else:
            unmatched.append(item.get("quote"))
    result["evidence_check"] = {"checked": checked, "matched": matched,
                                "unmatched": unmatched}
    result["evidence_verified"] = not unmatched
    if unmatched:
        result["classifications"]["review_status"] = "Needs review"

    # Allegation firewall: an organiser-band involvement score with ZERO verified
    # evidence is an unsupported "instigator" label — reject rather than emit.
    inv = result["scores"]["involvement"]["score"]
    if inv <= 4 and matched == 0:
        raise ValidationError(
            "involvement scored in the organiser band (0-4) with no verified "
            "supporting quote — every serious finding must cite the employee's own words")
    return result


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

def _resolve_employee(call):
    """Name from our own records — never from the model."""
    phone = (call.get("caller") or "").strip()
    name, resolved_from = "", ""
    cid = call.get("campaign_id")
    if phone and cid:
        try:
            import eo_db
            cc = eo_db.cc_find_recent_by_phone(phone)
            if cc and str(cc.get("campaign_id")) == str(cid) and cc.get("name"):
                name, resolved_from = str(cc["name"]), "campaign_contact"
        except Exception:
            pass
    if not name and phone:
        try:
            import directory
            rec = directory.record_for(phone)
            if rec.get("first_name"):
                name, resolved_from = rec["first_name"], "roster"
        except Exception:
            pass
    return {"name": name, "phone": phone,
            "employee_id": call.get("employee_id") or "",
            "resolved_from": resolved_from}


def _promote_flat_fields(call):
    """Headline assessment values as flat fields → picked up by the store index."""
    a = call.get("assessment") or {}
    call["assessment_status"] = a.get("status")
    call["assessment_attempts"] = a.get("attempts") or 0
    call["assessment_scored_at"] = a.get("scored_at")
    override = a.get("override") or {}
    call["assessment_reviewed"] = bool(override)
    if a.get("status") == "completed":
        call["assessment_score"] = a.get("total_score")
        call["assessment_red_flag"] = (override.get("red_flag_level")
                                       or (a.get("classifications") or {}).get("red_flag_level"))
        call["assessment_review_status"] = normalize_review_status(
            override.get("review_status")
            or (a.get("classifications") or {}).get("review_status"))
        call["assessment_involvement"] = (a.get("classifications") or {}).get("incident_involvement")
        call["assessment_summary"] = a.get("summary") or ""
    else:
        for k in ("assessment_score", "assessment_red_flag", "assessment_review_status",
                  "assessment_involvement", "assessment_summary"):
            call.pop(k, None)


async def _write_assessment(call_id, mutator):
    saved = await store.update_call(call_id, mutator)
    if saved is None:
        logger.warning(f"assessment write skipped: call {call_id} not found")
    return saved


async def run_assessment(call_id, trigger="auto", requested_by="system", force=False):
    """The scoring worker. A failure marks assessment.status=failed and never
    touches the rest of the call record."""
    async with _get_sem():
        call = await store.load_call(call_id)
        if call is None:
            logger.warning(f"assessment: call {call_id} not found")
            return None
        if not force:
            ok, reason = eligible(call)
            if not ok:
                logger.info(f"assessment: {call_id} ineligible ({reason})")
                return None
        elif not (call.get("transcript") or []):
            def _skip(c):
                c["assessment"] = {"status": "skipped", "version": RUBRIC_VERSION,
                                   "error": "no transcript", "trigger": trigger,
                                   "requested_by": requested_by,
                                   "human_review_required": True,
                                   "scored_at": _now_iso()}
                _promote_flat_fields(c)
            await _write_assessment(call_id, _skip)
            return None

        prior = call.get("assessment")

        def _mark_pending(c):
            hist = list((c.get("assessment") or {}).get("history") or [])
            p = c.get("assessment") or {}
            if p.get("status") == "completed":
                hist.append({"scored_at": p.get("scored_at"), "model": p.get("model"),
                             "total_score": p.get("total_score"),
                             "red_flag_level": (p.get("classifications") or {}).get("red_flag_level"),
                             "review_status": (p.get("classifications") or {}).get("review_status"),
                             "trigger": p.get("trigger"), "override": p.get("override")})
            c["assessment"] = {"status": "pending", "version": RUBRIC_VERSION,
                               "model": _model_name(), "trigger": trigger,
                               "requested_by": requested_by,
                               "attempts": int((p or {}).get("attempts") or 0),
                               "started_at": _now_iso(),
                               "human_review_required": True,
                               "history": hist}
            _promote_flat_fields(c)
        await _write_assessment(call_id, _mark_pending)

        max_attempts = _env_int("TT_ASSESSMENT_MAX_ATTEMPTS", 3)
        timeout_s = _env_float("TT_ASSESSMENT_TIMEOUT_S", 120.0)
        backoffs = [5, 15, 45]
        error = validator_error = ""
        result = tokens = None
        attempts_done = int((prior or {}).get("attempts") or 0)

        for attempt in range(max_attempts):
            attempts_done += 1
            try:
                raw, tokens = await asyncio.wait_for(
                    _generate(build_prompt(call, validator_error=validator_error)),
                    timeout=timeout_s)
                result = verify_evidence(validate_result(raw), call)
                error = ""
                break
            except (ValidationError, json.JSONDecodeError) as e:
                validator_error = str(e)
                error = f"validation: {e}"
                logger.warning(f"assessment {call_id} attempt {attempts_done} invalid: {e}")
            except asyncio.TimeoutError:
                error = f"timeout after {timeout_s:.0f}s"
                logger.warning(f"assessment {call_id} attempt {attempts_done} timed out")
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                logger.warning(f"assessment {call_id} attempt {attempts_done} failed: {e}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(backoffs[min(attempt, len(backoffs) - 1)])

        if result is None:
            def _mark_failed(c):
                a = c.get("assessment") or {}
                a.update({"status": "failed", "error": error, "attempts": attempts_done,
                          "scored_at": _now_iso(), "human_review_required": True})
                c["assessment"] = a
                _promote_flat_fields(c)
            await _write_assessment(call_id, _mark_failed)
            return None

        cost = pricing.compute_assessment_cost(tokens)
        employee = _resolve_employee(call)

        def _mark_done(c):
            a = c.get("assessment") or {}
            block = {
                "status": "completed", "version": RUBRIC_VERSION, "model": _model_name(),
                "scored_at": _now_iso(), "attempts": attempts_done, "error": None,
                "trigger": trigger, "requested_by": requested_by,
                "human_review_required": True,        # ALWAYS — the AI never decides
                "questions_total": QUESTIONS_TOTAL,
                "employee": employee,
                "tokens": tokens, "cost_usd": cost,
                "history": a.get("history") or [],
            }
            block.update(result)
            c["assessment"] = block
            _promote_flat_fields(c)
            total, estimated = pricing.compute_total(c)
            c["total_cost_usd"] = total
            c["cost_estimated"] = estimated
        saved = await _write_assessment(call_id, _mark_done)
        if saved is not None:
            logger.info(f"assessment {call_id}: score={result['total_score']}/100 "
                        f"red_flag={result['classifications']['red_flag_level']} "
                        f"review={result['classifications']['review_status']} "
                        f"evidence_verified={result['evidence_verified']} cost=${cost:.4f}")
        return saved


# ---------------------------------------------------------------------------
# Boot recovery sweep
# ---------------------------------------------------------------------------

async def sweep_once():
    """Re-enqueue interrupted ('pending') and missed/failed-under-max assessments."""
    if not _enabled():
        return 0
    max_attempts = _env_int("TT_ASSESSMENT_MAX_ATTEMPTS", 3)
    listing = await store.list_calls({})
    queued = 0
    for m in listing.get("items") or []:
        status = m.get("assessment_status")
        if status in ("completed", "skipped"):
            continue
        if status == "failed" and (m.get("assessment_attempts") or 0) >= max_attempts:
            continue
        if status is None:
            call = await store.load_call(m.get("id"))
            ok, _reason = eligible(call or {})
            if not ok:
                continue
        schedule(m["id"], trigger="sweep", requested_by="system")
        queued += 1
    if queued:
        logger.info(f"assessment sweep queued {queued} call(s)")
    return queued


async def sweep_loop():
    """One immediate sweep at boot, then hourly (covers 'call ended while the
    Gemini API was down'). Started from main.py's startup hook."""
    try:
        await asyncio.sleep(5)                # let store.init() finish populating
        while True:
            try:
                await sweep_once()
            except Exception as e:
                logger.warning(f"assessment sweep failed: {e}")
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
