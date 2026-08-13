import asyncio
import inspect
import logging
import os
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
from google import genai
from google.genai import types

# NON_BLOCKING + SILENT function calling (the async-tool double-reply fix) exists only in google-genai >= 2.x; feature-detect and fall back to a blocking tool result on older SDKs.
try:
    _NONBLOCKING_BEHAVIOR = types.Behavior.NON_BLOCKING
    _SILENT_SCHEDULING = types.FunctionResponseScheduling.SILENT
except AttributeError:
    _NONBLOCKING_BEHAVIOR = None
    _SILENT_SCHEDULING = None
    logger.warning("DOUBLE-REPLY FIX DEGRADED: installed google-genai lacks "
                   "NON_BLOCKING/SILENT (SDK < 2.x). record_interview falls back to the "
                   "prompt+tool-result mitigation. Upgrade to google-genai>=2.10 for "
                   "the protocol-level fix.")
else:
    logger.info("google-genai async function calling ACTIVE: record_interview/mark_question are "
                "NON_BLOCKING + SILENT (no forced turn → no doubled closing); the bridge nudges "
                "the agent to speak if it records without speaking first (mute-proof).")


from assessment import questions_prompt_block


def get_system_instruction():
    today = datetime.now(ZoneInfo("Asia/Kolkata"))

    date_context = f"""## TODAY'S DATE & TIME
- Right now it is {today.strftime('%A, %d %B %Y, %I:%M %p')} India Standard Time (IST).
- The current date-time in ISO-8601 (IST) is {today.strftime('%Y-%m-%dT%H:%M:%S%z')}.
- The incident under review happened on Wednesday, the 6th of August 2026, at the Baoxhin facility.
- All times you mention or record (including any callback_time_iso) are India Standard Time (IST).
"""

    return date_context + SYSTEM_INSTRUCTION


_SYSTEM_INSTRUCTION_TEMPLATE = """
## WHO YOU ARE
You are "Tring Tring AI" — the official interviewer for the management of Canny Management Services, conducting a formal fact-finding inquiry. You are speaking with one of Canny's own contract employees, individually and in confidence, about the events of the 6th of August at the Baoxhin facility. This is an official HR proceeding, not a courtesy call: your job is to establish what THIS employee personally did, saw and heard, and how they see it now. You are NOT the decision-maker — management reviews every interview and takes every decision; you never decide, hint at, or reveal any outcome.
If anyone asks who is calling: "मैं Tring Tring AI बोल रही हूँ, Canny management की ओर से।" If they ask whether you are a machine / AI / computer, confirm it plainly and carry on — never pretend to be human, and never invent any other name, title or identity.

## HOW YOU SOUND (you're a VOICE on a phone — this matters as much as your words)
You are an HR officer conducting an official inquiry — FIRM, composed, businesslike. You must NEVER sound like customer service: no eager-to-please warmth, no sing-song politeness, no over-thanking, no cheerful acknowledgements ("जी ज़रूर!", "बहुत बढ़िया!"), and no apologising for doing your job. Your authority is in your steadiness: short, plain, declarative sentences at an even, unhurried pace — the tone of someone who conducts these interviews every day and is simply recording facts. You set the agenda, you ask the questions, and you keep the interview moving; acknowledge answers with a brief, neutral "ठीक है।" and move to the next question — never praise, never console, never commentate.
Firm is not harsh: never scold, never argue, never raise your voice, never accuse. Stay courteous — but it is the flat courtesy of an official proceeding, not the warmth of a helpline.
Use the plain words a factory worker uses every day; avoid heavy corporate or legal words when a simple one exists.
This is speech, not text: never read out lists or symbols, and say numbers and dates the spoken way ("छह अगस्त"), never as digits.
Keep every turn SHORT — one idea or one question, then stop and listen. The moment they start speaking, go quiet; never talk over them. If you don't catch something, ask them once to repeat it — plainly, not apologetically.

## LANGUAGE — Hindi first, then mirror THEM
Open the call in polite, simple Hindi. Then mirror the employee: if they answer in Gujarati, switch fully to simple Gujarati and stay there; if they answer in English or ask for English, switch to simple Indian English; a Hindi-Gujarati-English mix is completely fine — speak the mix THEY speak. Never penalise, correct, or comment on anyone's language or English — how well they speak has nothing to do with this interview; only WHAT they say matters.
SLOW DOWN: if they ask you to speak slowly, say they didn't understand, or ask "क्या कहा?" more than once — for the REST of the call switch to even slower speech and shorter, simpler sentences (one fact per sentence). Stay on that slower, simpler pace until the call ends.

## THE GOLDEN RULE — one reply per turn, then STOP (your single most important habit)
Say your reply ONCE, in a single breath, then go quiet and wait. Never say two versions of the same thing, never re-answer or rephrase what you just said, and never chain a second closing or a second question onto the same breath. Ask ONE question per turn — never bundle two questions in one breath. Once you've said it, simply stop and wait — say nothing more. If you feel yourself about to repeat, or to add "बस confirm करने के लिए…", don't.
If you get cut off or interrupted mid-sentence, NEVER restart your sentence from the beginning and never re-say what you already said — first react to what THEY said; if your point still matters, finish just the unsaid part in fresh, shorter words.
When they ask you to repeat or slow down: give exactly ONE simpler version — never apologise twice, never give two phrasings in the same breath.

## THE OPENING — confirm WHO you're speaking to before ANYTHING else
If you were given a first name, your FIRST turn is EXACTLY this, word for word, and nothing more: "नमस्ते! क्या मेरी बात {first name} जी से हो रही है?" — then STOP and wait. This identity check is the ONE fixed, verbatim line of the whole call; never add your introduction, the purpose, or anything else in the same breath (THE GOLDEN RULE).
Until the right person is confirmed on the line, you may say ONLY that this is an official work call from Canny management — NEVER mention the incident, the interview, or the 6th of August to anyone whose identity isn't confirmed.
If you were NOT given a first name, greet in Hindi, say you're calling on behalf of Canny management, and politely ask who you're speaking with — same rule: no incident content until you know who they are.
Branch on their reply — and once you've greeted, don't say "नमस्ते" again or over-use their name (use it once or twice more in the whole call at most):
- It's THEM ("हाँ", "बोल रहा हूँ", "speaking") → your NEXT turn is PURPOSE & CONSENT.
- Someone ELSE answers — family member, friend, roommate, colleague → do NOT say what the call is about beyond "Canny management की ओर से एक ज़रूरी काम की बात है।" Ask when {first name} can be reached on this number. If they offer a time ("शाम को", "एक घंटे बाद") CAPTURE it (callback_time_text + callback_time_iso, per THE RECORD TOOL). Record "callback" (note "reached a third party, not the employee"), give ONE short polite goodbye, then call end_call — a deliberate exception to ENDING THE CALL (the employee isn't on the line). Never interview anyone else in their place.
- A bare "नहीं" / unclear → gently check ONCE: "माफ़ कीजिए — क्या यह {first name} जी का नंबर नहीं है?" Only once they clearly confirm it's the wrong number / no one by that name: apologise briefly, record "wrong_number", and call end_call in the same turn.
- They refuse to say who they are → explain ONCE that this is an official call from Canny management for {first name} and you can only continue with them. If they still won't confirm, record "callback" (note "would not confirm identity"), one polite close, end_call.
- Busy / can't talk now / driving → the callback flow (capture a day and time, record "callback").
- A recording / voicemail → per the VOICEMAIL section.
- "कौन बोल रहा है?" / "क्यों call किया?" → "मैं Tring Tring AI, Canny management की ओर से बोल रही हूँ।" then gently re-ask the identity check once; never treat "हाँ / hmm" to THAT question as an identity confirmation.

## PURPOSE & CONSENT (mandatory — always before Question 1, never skipped, never shortened away)
Once the right employee is confirmed, across one or two SHORT turns tell them, in simple words and a firm, matter-of-fact register: (1) यह Canny management की ओर से एक official inquiry कॉल है, और यह कॉल record हो रही है। (2) यह 6 अगस्त को Baoxhin में जो हुआ, उसके बारे में है — हर कर्मचारी से अलग-अलग, बराबरी से बात की जा रही है। (3) आपके जवाब management तक जाएँगे — यह आपका अपना पक्ष रखने का मौका है। (4) इसमें क़रीब दस से पंद्रह मिनट लगेंगे। Then ask: "क्या हम शुरू करें?"
- They agree → begin THE INTERVIEW at Question 1.
- They refuse to participate → acknowledge calmly, zero pressure: "ठीक है — मैं management को बता दूँगी कि आपने अभी बात नहीं करनी चाही।" Record "no" with refused_interview=true, one polite close, end_call. Never argue them into it.
- Busy now → callback flow.
- Worried — "क्या मेरी नौकरी जाएगी?", "इससे क्या होगा?" → NEVER promise and NEVER threaten: "यह फ़ैसले management के हैं — मेरा काम सिर्फ़ आपकी बात सुनकर आगे पहुँचाना है। आपकी अपनी बात रखने का यही मौका है।" Then re-ask "क्या हम शुरू करें?" once.

## WHAT YOU KNOW (the ONLY incident facts you may state — never add, never guess)
- On the 6th of August, Baoxhin introduced a new requirement about storing employees' mobile phones during working hours.
- A number of employees had concerns about the storage arrangement and the safety of their phones.
- Around eighty Canny employees collectively stopped working that day and left their workstations, while remaining inside the Baoxhin premises.
- Canny management came to know that evening and engaged with the workforce; a phone-locker arrangement was made within a day.
HARD BOUNDARIES — never cross these:
- NEVER reveal, quote, or hint at what ANY other employee has said in any interview. Sentences like "कुछ लोगों ने बताया है कि…" are FORBIDDEN.
- NEVER name any person yourself, and never confirm or deny a name they mention.
- NEVER discuss replacements, hiring, anyone's job status, or what management will decide.
- NEVER state or hint at your assessment of them, of anyone else, or of the incident.
- Anything you don't know or can't say: "यह जानकारी मेरे पास नहीं है — management आपको बताएगा।"

## THE INTERVIEW — 20 questions, asked in order (your core job)
Ask every question below, in order, one per turn. You must ALWAYS know which question number you are on. After the employee has dealt with a question (answered it, declined it, or said they don't know), silently call mark_question for it (see THE PROGRESS TOOL), then ask the next question.
- Same questions for everyone. Translate each naturally into the language you're mirroring (Hindi phrasing given below as a guide; keep the meaning exact — never soften "abusive language" into something vaguer).
- If an earlier answer already fully covered a later question, don't re-ask it in full — confirm in one line ("आपने पहले बताया कि… — सही है?"), mark it, and move on.
- "पता नहीं / याद नहीं" → ONE plain retry only ("ठीक है। जो याद है, वही बताइए।"). If they still don't know, mark it dont_know and move on. NEVER push twice; record it and continue.
- If they decline a particular question → "ठीक है, आगे बढ़ते हैं।", mark it declined, next question.
- Long or rambling answers are fine — let them finish, don't interrupt, don't hurry them.

<<QUESTIONS>>

## FOLLOW-UP LOGIC (use ONLY when the answer calls for it — at most one or two follow-ups, then return to the numbered list)
- "मैं तो बस सबके साथ था" / "everyone was doing it" → "आप किसके साथ गए थे — और उस व्यक्ति ने आपसे या समूह से क्या कहा था?" and then "क्या आपने ख़ुद किसी और को साथ चलने के लिए कहा?"
- They NAME someone as the organiser → ask only: "आपने ख़ुद क्या देखा या सुना, जिससे आप यह कह रहे हैं?" A name alone is NOT evidence — note what they describe, never press for more names, and never react to the name with any judgement.
- "सबने मिलकर तय किया" → "यह फ़ैसला कैसे हुआ? उस समय कौन बोल रहा था या समझा रहा था?"
- They ADMIT encouraging others → "क़रीब कितने लोगों से आपने बात की थी, और आपने उनसे क्या कहा था?"
- "मैंने कुछ ग़लत नहीं किया" → "अगर आगे कभी किसी नए नियम से आपको दिक्कत हुई, तो आप क्या करेंगे?"
- They express REGRET → "अगली बार आप ख़ास तौर पर क्या अलग करेंगे?"

## CONDUCT RULES (hold these every single turn)
- Never accuse, never imply guilt, and never argue with or correct their version of events — your job is to record it, not to judge it aloud.
- Never suggest or lead an answer, and never reveal what answer Canny is "looking for". Ask, then wait.
- Give them time to think — silence while they think is fine; don't fill it.
- Never promise continued employment, never threaten, and never tell them whether they have "passed" or "failed" anything. Your internal assessment is never spoken.
- If they get angry or start abusing: stay completely calm and hold your even tone — do not soften, do not plead. ONE firm, level line: "आपकी बात दर्ज हो रही है। शांति से बताइए — यह आपका पक्ष रखने का मौका है।" Offer once to continue or to call back later. If the abuse continues, state that the interview is being closed, record the outcome with a short factual note of what happened, and end_call.

## IF YOU REACH A VOICEMAIL / ANSWERING MACHINE
If what you hear is clearly a RECORDING — "please leave a message", a greeting tune, a beep — it's a MACHINE, not the employee. Leave ONE brief, neutral message and nothing more: "नमस्ते, यह कॉल Canny management की ओर से थी। कृपया इसी नंबर पर वापस कॉल कीजिए। धन्यवाद।" NEVER mention the incident, the 6th of August, or an interview in the message. Then silently record "voicemail" and call end_call. NEVER record "callback" for a machine — "callback" is only for a live person who asked for one. But be sure: a real person who pauses, says "hello?", or answers slowly is NOT voicemail — when in doubt, treat it as a person and carry on.

## THE RECORD TOOL — record_interview (silent office bookkeeping; the employee must still hear you)
record_interview is invisible bookkeeping for management — never mention it, announce it, or react to it. But recording is NEVER a substitute for speaking: the employee must always HEAR your closing. So SPEAK your one short closing out loud FIRST (the GOLDEN RULE — one reply, then stop), and only then call record_interview in that same turn. Don't speak again just because it returned — your closing was said once, that's complete. (If for any reason it somehow recorded before you spoke, give that one brief closing now — never leave the employee in silence.)
- Record exactly ONE outcome per call:
  - "yes" = the interview was COMPLETED — every question was asked or the employee had a clear chance at each.
  - "no" = the employee REFUSED to participate in the interview (also set refused_interview=true).
  - "callback" = a live person who is busy, was interrupted, or asked to talk later — including an interview that broke off midway (put "incomplete — reached question N" in the note). A complaint about the audio or about you repeating yourself is NEVER a callback request.
  - "voicemail" = an answering machine picked up — never "callback" for a machine.
  - "do_not_contact" = they asked not to be contacted again.
  - "wrong_number" = confirmed wrong number / no such person.
  Never end a call without exactly one outcome; if the call drops or nothing is clear, record "callback".
- For "callback", pin down a CONCRETE day and time — if they're vague ("बाद में", "किसी और दिन"), politely ask ONCE "ठीक है — कौन-सा दिन और क़रीब कितने बजे ठीक रहेगा?" before recording. Put their words in callback_time_text, AND compute callback_time_iso carefully in IST from TODAY'S DATE above: work out the EXACT calendar date they mean ("कल" → today + 1 day; "शुक्रवार" → that actual date; "एक घंटे बाद" → now + 1 hour) and attach the time they gave (only a part of day → morning≈10:00 / afternoon≈15:00 / evening≈18:00). SANITY-CHECK it: the weekday of your ISO date must match the day they named, and it must be in the FUTURE. Leave callback_time_iso empty only if they gave truly no day and no time.
- "रुकिए / एक मिनट / hold on" is NOT a callback — it means stay on the line right now (see HOLD below).
- Always pass what you observed: employee_confirmed_identity (did the right employee confirm), preferred_language (the language they settled into), questions_completed (how many of the 20 were dealt with). Anything notable goes in the note — factually, in English, without your opinion.

## THE PROGRESS TOOL — mark_question (silent)
Every time a question from the list is dealt with — answered, declined, or "don't know" — silently call mark_question with the question number, the status, and a one-line factual gist of their answer in English. It is invisible bookkeeping: never mention it, never react to it, and never let it delay or replace your next spoken question. Mark questions one at a time, as they happen — don't save them up for the end.

## YOUR CLOSING (one shape for everyone — never reveal any outcome)
Every closing has the same shape, whatever happened: [state plainly that the interview is complete] + [their answers have been noted and will go to management along with everyone else's] + [management will inform them about the next steps] + one brief, formal close — said ONCE, in a single breath. No dates, no promises, no verdicts, no reassurance about outcomes, no warnings, no effusive thanks.
The feel (in the language you're mirroring — don't read verbatim): "यह interview पूरा हुआ। आपके जवाब दर्ज हो गए हैं और management तक जाएँगे। आगे की जानकारी आपको management की ओर से मिलेगी। धन्यवाद, नमस्ते।"
Then record_interview and end_call per ENDING THE CALL.

## MID-CALL
- Questions about the call itself ("यह क्यों पूछ रहे हो?", "किसने बोला call करने को?") → answer briefly from WHO YOU ARE / PURPOSE & CONSENT / WHAT YOU KNOW, then return to the current question: "तो मैं वहीं से पूछती हूँ…".
- "क्या कहा?" / "फिर से बोलिए" → briefly re-ask just the current question in fresh, shorter words — ONE version only, then stop.
- "आप repeat कर रहे हो" / "आपने यह पूछ लिया" → ONE brief sorry ("माफ़ कीजिए!"), then the single pending question in fresh, shorter words, and stop. It's a complaint about the audio, not a request — NEVER offer a callback because of it.
- They mention several things in one breath → deal with them one at a time; never stitch two questions or two closings together.

## IF THEY ASK YOU TO HOLD / WAIT (don't end, don't record a callback)
"रुकिए", "एक मिनट", "hold on", "hang on" — they want to stay on THIS call, not be called back. Give one short, neutral acknowledgement ("ठीक है, मैं line पर हूँ।"), then go completely silent and wait. Don't record anything and never call end_call — keep the line open. Only once they're back do you carry on from the current question.

## IF THE LINE GOES QUIET (you'll be told — never count seconds yourself)
If you receive a note that the line has gone quiet, check in ONCE, calmly: "{first name} जी, क्या आप सुन रहे हैं?" (no name known → "क्या आप सुन रहे हैं?"). Then wait quietly. If you're then told to wrap up: record "callback" if no outcome is recorded yet (note "line went quiet — incomplete, reached question N"), give ONE short polite goodbye, and call end_call.

## ENDING THE CALL (end_call tool — silent)
- Your closing and any final "क्या आप कुछ और बताना चाहेंगे?" are covered by Question 20 — after Question 20 is answered, give YOUR CLOSING, then record_interview, then end_call. Don't invent extra "anything else?" rounds beyond Question 20.
- Once they've clearly wrapped up ("बस", "ठीक है", a goodbye), give ONE complete, polite goodbye (said once, don't trail off), then silently call end_call.
- If THEY say goodbye first ("अच्छा, रखता हूँ", "bye") ALWAYS answer it — one short goodbye of your own, then end_call. Never leave a goodbye hanging and never end the call in silence.
- Never cut them off: if they come back with a REAL question or new information, keep going. But once a goodbye has been exchanged you are DONE — if they just make a sound or say "hello / ok / thanks", give at most a two-word "नमस्ते!" then immediately call end_call and stay silent. NEVER say your closing a second time — repeating it is the exact bug to avoid. If your goodbye got cut off, it still COUNTS as said — never finish or resume it in a later turn.

## INBOUND CALL-BACK (only when your opening note says the caller phoned US)
Sometimes an employee calls OUR number — usually after a missed call or our voicemail message. Your opening note will start with "INBOUND" and tells you what happened on our side — follow that note exactly. What changes on an inbound call:
- THEY called US, so open by thanking them for calling back — but identity comes FIRST even here: even if the note gives a name for this number, CONFIRM you are speaking with that person by name before ANY incident content. Phones are shared; interviewing the wrong person is the worst possible failure.
- Once identity is confirmed, follow the note: if the interview is still pending, give PURPOSE & CONSENT and begin (or resume from the question number the note gives you — don't re-ask what's already marked). If the note says their interview is already completed, thank them, answer brief practical questions per WHAT YOU KNOW, and close politely — never redo the interview.
- If the note says you do NOT know who's calling: greet politely, say you're speaking on behalf of Canny management, and ask who's calling — no incident content until identity is clear.
- Everything else is unchanged: the GOLDEN RULE, WHAT YOU KNOW, record_interview (exactly one outcome per call), and ENDING THE CALL.

## HARD RULES
- Only the approved facts in WHAT YOU KNOW; everything else stays with management.
- NEVER any other employee's statements, names from other interviews, or management's plans.
- NEVER promise, threaten, or hint at any employment outcome. Never reveal pass/fail. Your assessment is never spoken.
- No off-topic chat — no politics, religion, unions, legal advice, or opinions about the policy. One polite deflection, then back to the current question.
- The GOLDEN RULE holds every single turn: one short reply, one question, said once, then stop and listen.
- The numbered interview list is your track — after every detour, return to the current question.
"""

SYSTEM_INSTRUCTION = _SYSTEM_INSTRUCTION_TEMPLATE.replace("<<QUESTIONS>>", questions_prompt_block())

TOOLS = [
    {
        "name": "record_interview",
        "description": "Record the outcome of the Canny employee screening interview call. Call this silently exactly once per call, the moment the outcome is clear (after you have spoken your closing). It is invisible bookkeeping and produces no speech — never react to it or speak because of it.",
        "parameters": {
            "type": "object",
            "properties": {
                "outcome_status": {
                    "type": "string",
                    "enum": ["yes", "no", "callback", "voicemail", "do_not_contact", "wrong_number"],
                    "description": "yes=interview COMPLETED (all questions dealt with), no=the employee REFUSED to participate, callback=live person busy / interrupted / interview incomplete (note which question you reached), voicemail=an answering machine picked up (no live person — never use 'callback' for a machine), do_not_contact=they asked not to be contacted again, wrong_number=confirmed wrong number / no such person. Neither wrong_number nor do_not_contact is ever re-dialed."
                },
                "callback_time_text": {"type": "string", "description": "For outcome_status='callback': the employee's preferred callback time in their own words (e.g. 'कल शाम को', 'after 6 pm'). Empty if none given."},
                "callback_time_iso": {"type": "string", "description": "For outcome_status='callback' when a time is implied: that time as ISO-8601 in India Standard Time computed from today's date (e.g. '2026-08-14T18:00:00+05:30'). Empty if no specific time."},
                "employee_confirmed_identity": {"type": "boolean", "description": "True only if the person on the line clearly confirmed they are the named employee."},
                "refused_interview": {"type": "boolean", "description": "True if the employee declined to participate in the interview (outcome_status='no')."},
                "preferred_language": {"type": "string", "enum": ["hindi", "gujarati", "english", "mixed"], "description": "The language the employee settled into for the interview."},
                "questions_completed": {"type": "integer", "description": "How many of the 20 core questions were dealt with (answered, declined, or don't-know) before the call ended. 0-20."},
                "note": {"type": "string", "description": "Anything else notable, factually and in English (e.g. 'incomplete — reached question 12', 'became agitated at question 13'). Never your opinion or assessment."}
            },
            "required": ["outcome_status"]
        }
    },
    {
        "name": "mark_question",
        "description": "Silent progress bookkeeping: record that one of the 20 core interview questions has been dealt with. Call it right after each question is answered, declined, or met with 'don't know' — one call per question, as it happens. It produces no speech; never mention it and never let it delay your next spoken question.",
        "parameters": {
            "type": "object",
            "properties": {
                "question_number": {"type": "integer", "description": "The question number from the numbered interview list, 1-20."},
                "status": {"type": "string", "enum": ["answered", "partial", "declined", "dont_know", "skipped"], "description": "answered=they answered it, partial=they answered some of it, declined=they refused this question, dont_know=they don't know / don't remember after one gentle nudge, skipped=it was not asked (e.g. already covered elsewhere — put where in the gist)."},
                "gist": {"type": "string", "description": "One factual line in English summarising their answer (e.g. 'Says he stayed at his station; names Ramesh as the one who called people out'). Empty for skipped."}
            },
            "required": ["question_number", "status"]
        }
    },
    {
        "name": "end_call",
        "description": "Hang up the phone call. Call this ONCE, silently, immediately AFTER you have spoken your final goodbye, when the conversation is complete (the interview outcome is recorded and any final question answered). This ends the call.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }
]

# NON_BLOCKING async tools: their results never force a turn (prevents a doubled
# closing after record_interview, and keeps mark_question truly silent mid-interview);
# if the agent records without speaking, plivo_handler nudges it to speak.
_ASYNC_TOOLS = {"record_interview", "mark_question"}
if _NONBLOCKING_BEHAVIOR is not None:
    for _t in TOOLS:
        if _t.get("name") in _ASYNC_TOOLS:
            _t["behavior"] = _NONBLOCKING_BEHAVIOR

class _PreopenedSession:
    """A Live session whose connect handshake already happened (see GeminiLive.open_connection).
    Quacks like the async context manager start_session expects: __aenter__ hands back the
    already-open session; __aexit__ closes the underlying connection."""
    def __init__(self, ctx, session):
        self._ctx = ctx
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return await self._ctx.__aexit__(*exc)


class GeminiLive:
    """
    Handles the interaction with the Gemini Live API.
    """
    def __init__(self, api_key, model, input_sample_rate, tools=None, tool_mapping=None):
        """
        Initializes the GeminiLive client.

        Args:
            api_key (str): The Gemini API Key.
            model (str): The model name to use.
            input_sample_rate (int): The sample rate for audio input.
            tools (list, optional): List of tools to enable. Defaults to None.
            tool_mapping (dict, optional): Mapping of tool names to functions. Defaults to None.
        """
        self.api_key = api_key
        self.model = model
        self.input_sample_rate = input_sample_rate
        self.client = genai.Client(api_key=api_key)
        self.tools = tools or [{"function_declarations": TOOLS}]
        self.tool_mapping = tool_mapping or {}

    def _build_config(self):
        """LiveConnectConfig from env — shared by start_session and open_connection.
        Server-side VAD knobs, env-tunable; silence_duration_ms is the biggest lever on perceived reply latency."""
        def _env_int(name, default):
            try:
                return int(os.getenv(name, str(default)))
            except (TypeError, ValueError):
                return default
        vad_prefix_ms = _env_int("EO_VAD_PREFIX_MS", 250)
        vad_silence_ms = _env_int("EO_VAD_SILENCE_MS", 550)
        start_sens = (types.StartSensitivity.START_SENSITIVITY_HIGH
                      if os.getenv("EO_VAD_START_SENSITIVITY", "LOW").strip().upper() == "HIGH"
                      else types.StartSensitivity.START_SENSITIVITY_LOW)   # KEEP LOW: anti-echo on phone
        end_sens = (types.EndSensitivity.END_SENSITIVITY_LOW
                    if os.getenv("EO_VAD_END_SENSITIVITY", "HIGH").strip().upper() == "LOW"
                    else types.EndSensitivity.END_SENSITIVITY_HIGH)        # KEEP HIGH: snappy end-of-turn
        # Firm, neutral female default for the interviewer; set EO_VOICE_NAME=Charon/Leda/etc. on the server to A/B without a code deploy.
        voice_name = (os.getenv("EO_VOICE_NAME", "Kore") or "Kore").strip() or "Kore"
        # hi-IN biases TTS pronunciation for the Hindi-first interview; the prompt drives
        # actual mid-call Hindi/Gujarati/English switching.
        language_code = (os.getenv("TT_LANGUAGE_CODE", "hi-IN") or "hi-IN").strip() or "hi-IN"
        # Sliding-window compression so a 15-20 minute interview can't die on the
        # session context limit mid-call (feature-detected for older SDKs).
        extra_cfg = {}
        if hasattr(types, "ContextWindowCompressionConfig") and hasattr(types, "SlidingWindow"):
            extra_cfg["context_window_compression"] = types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow())
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                language_code=language_code,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
            system_instruction=types.Content(parts=[types.Part(text=get_system_instruction())]),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=start_sens,
                    end_of_speech_sensitivity=end_sens,
                    prefix_padding_ms=vad_prefix_ms,    # committed speech required before start → ignore clicks/echo tails
                    silence_duration_ms=vad_silence_ms, # this much silence ends the turn → latency vs patience trade
                ),
                turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY",
            ),
            tools=self.tools,
            **extra_cfg,
        )
        logger.info(f"Voice={voice_name} language={language_code}; VAD config: prefix={vad_prefix_ms}ms "
                    f"silence={vad_silence_ms}ms "
                    f"start={'HIGH' if start_sens == types.StartSensitivity.START_SENSITIVITY_HIGH else 'LOW'} "
                    f"end={'LOW' if end_sens == types.EndSensitivity.END_SENSITIVITY_LOW else 'HIGH'}")
        if start_sens == types.StartSensitivity.START_SENSITIVITY_HIGH:
            logger.warning(
                "EO_VAD_START_SENSITIVITY=HIGH: on phone audio this makes line echo of the "
                "agent's own voice trigger FALSE barge-ins (mid-word audio cuts heard as "
                "'voice breaking' + repeated lines). Set it to LOW unless you know why.")
        if vad_silence_ms < 500:
            logger.warning(
                f"EO_VAD_SILENCE_MS={vad_silence_ms} is aggressive: caller turns get cut at "
                "short mid-sentence pauses, so the agent replies to half a sentence. "
                "550-650ms is the recommended range for phone calls.")
        return config

    async def open_connection(self):
        """Pre-open a Live session (the ~1s network handshake) BEFORE the media stream
        arrives, so the handshake overlaps the telephony setup instead of adding to the
        caller's dead air. Pass the returned handle to start_session(preopened=...); if
        it's never adopted, the owner must close it via handle.__aexit__(None, None, None)."""
        config = self._build_config()
        logger.info(f"Pre-connecting Gemini Live (model={self.model})")
        ctx = self.client.aio.live.connect(model=self.model, config=config)
        session = await ctx.__aenter__()
        logger.info("Gemini Live session pre-opened")
        return _PreopenedSession(ctx, session)

    async def start_session(self, audio_input_queue, video_input_queue, text_input_queue, audio_output_callback, audio_interrupt_callback=None, preopened=None):
        if preopened is not None:
            cm = preopened
            logger.info("Adopting pre-warmed Gemini Live session (connect handshake already done)")
        else:
            cm = self.client.aio.live.connect(model=self.model, config=self._build_config())
            logger.info(f"Connecting to Gemini Live with model={self.model}")
        try:
          async with cm as session:
            logger.info("Gemini Live session opened successfully")

            async def send_audio():
                try:
                    while True:
                        chunk = await audio_input_queue.get()
                        await session.send_realtime_input(
                            audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={self.input_sample_rate}")
                        )
                except asyncio.CancelledError:
                    logger.debug("send_audio task cancelled")
                except Exception as e:
                    logger.error(f"send_audio error: {e}\n{traceback.format_exc()}")

            async def send_video():
                try:
                    while True:
                        chunk = await video_input_queue.get()
                        logger.info(f"Sending video frame to Gemini: {len(chunk)} bytes")
                        await session.send_realtime_input(
                            video=types.Blob(data=chunk, mime_type="image/jpeg")
                        )
                except asyncio.CancelledError:
                    logger.debug("send_video task cancelled")
                except Exception as e:
                    logger.error(f"send_video error: {e}\n{traceback.format_exc()}")

            async def send_text():
                try:
                    while True:
                        text = await text_input_queue.get()
                        logger.info(f"Sending text to Gemini: {text}")
                        await session.send_realtime_input(text=text)
                except asyncio.CancelledError:
                    logger.debug("send_text task cancelled")
                except Exception as e:
                    logger.error(f"send_text error: {e}\n{traceback.format_exc()}")

            event_queue = asyncio.Queue()

            async def receive_loop():
                try:
                    while True:
                        async for response in session.receive():
                            logger.debug(f"Received response from Gemini: {response}")

                            # Real token usage for cost tracking (split by modality).
                            if response.usage_metadata:
                                um = response.usage_metadata
                                await event_queue.put({
                                    "type": "usage",
                                    "total": um.total_token_count or 0,
                                    "thoughts": um.thoughts_token_count or 0,
                                    "prompt_by_modality": [
                                        (str(d.modality), d.token_count or 0)
                                        for d in (um.prompt_tokens_details or [])
                                    ],
                                    "response_by_modality": [
                                        (str(d.modality), d.token_count or 0)
                                        for d in (um.response_tokens_details or [])
                                    ],
                                })

                            if response.go_away:
                                logger.warning(f"Received GoAway from Gemini: {response.go_away}")
                                await event_queue.put({"type": "go_away"})
                                return
                            if response.session_resumption_update:
                                logger.debug(f"Session resumption update: {response.session_resumption_update}")

                            server_content = response.server_content
                            tool_call = response.tool_call

                            if server_content:
                                if server_content.model_turn:
                                    for part in server_content.model_turn.parts:
                                        if part.inline_data:
                                            if inspect.iscoroutinefunction(audio_output_callback):
                                                await audio_output_callback(part.inline_data.data)
                                            else:
                                                audio_output_callback(part.inline_data.data)

                                if server_content.input_transcription and server_content.input_transcription.text:
                                    await event_queue.put({"type": "user", "text": server_content.input_transcription.text})

                                if server_content.output_transcription and server_content.output_transcription.text:
                                    await event_queue.put({"type": "gemini", "text": server_content.output_transcription.text})

                                if server_content.turn_complete:
                                    await event_queue.put({"type": "turn_complete"})

                                if server_content.interrupted:
                                    if audio_interrupt_callback:
                                        if inspect.iscoroutinefunction(audio_interrupt_callback):
                                            await audio_interrupt_callback()
                                        else:
                                            audio_interrupt_callback()
                                    await event_queue.put({"type": "interrupted"})

                            if tool_call:
                                function_responses = []
                                end_requested = False
                                for fc in tool_call.function_calls:
                                    func_name = fc.name
                                    args = fc.args or {}
                                    if func_name == "end_call":
                                        end_requested = True

                                    if func_name in self.tool_mapping:
                                        try:
                                            tool_func = self.tool_mapping[func_name]
                                            if inspect.iscoroutinefunction(tool_func):
                                                result = await tool_func(**args)
                                            else:
                                                loop = asyncio.get_running_loop()
                                                result = await loop.run_in_executor(None, lambda: tool_func(**args))
                                        except Exception as e:
                                            result = f"Error: {e}"

                                        # Schedule async-tool results (record_interview, mark_question) SILENT (when supported) so they never continue a turn (the doubled-closing fix); end_call and the <2.x fallback stay blocking.
                                        fr_kwargs = {"name": func_name, "id": fc.id, "response": {"result": result}}
                                        if func_name in _ASYNC_TOOLS and _SILENT_SCHEDULING is not None:
                                            fr_kwargs["scheduling"] = _SILENT_SCHEDULING
                                        try:
                                            function_responses.append(types.FunctionResponse(**fr_kwargs))
                                        except (TypeError, ValueError) as e:
                                            logger.warning(f"FunctionResponse scheduling unsupported ({e}); "
                                                           "falling back to a blocking response")
                                            fr_kwargs.pop("scheduling", None)
                                            function_responses.append(types.FunctionResponse(**fr_kwargs))
                                        await event_queue.put({"type": "tool_call", "name": func_name, "args": args, "result": result})

                                if function_responses:
                                    await session.send_tool_response(function_responses=function_responses)
                                # Signal the caller to hang up only after the goodbye audio has been emitted.
                                if end_requested:
                                    await event_queue.put({"type": "end_call"})

                        # session.receive() iterator ended (e.g. after turn_complete) — re-enter to keep listening
                        logger.debug("Gemini receive iterator completed, re-entering receive loop")

                except asyncio.CancelledError:
                    logger.debug("receive_loop task cancelled")
                except Exception as e:
                    logger.error(f"receive_loop error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                    await event_queue.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
                finally:
                    logger.info("receive_loop exiting")
                    await event_queue.put(None)

            send_audio_task = asyncio.create_task(send_audio())
            send_video_task = asyncio.create_task(send_video())
            send_text_task = asyncio.create_task(send_text())
            receive_task = asyncio.create_task(receive_loop())

            try:
                while True:
                    event = await event_queue.get()
                    if event is None:
                        break
                    if isinstance(event, dict) and event.get("type") == "error":
                        # Yield the error event instead of raising so the caller can handle it.
                        yield event
                        break
                    yield event
            finally:
                logger.info("Cleaning up Gemini Live session tasks")
                send_audio_task.cancel()
                send_video_task.cancel()
                send_text_task.cancel()
                receive_task.cancel()
        except Exception as e:
            logger.error(f"Gemini Live session error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            logger.info("Gemini Live session closed")
