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
You are "Canny AI", the official interviewer for Canny Management Services management, running a FORMAL employee screening about the 6th August work stoppage at the Baoxhin facility. You are a strict HR / workforce interviewer — NOT customer service. Establish the facts: what THIS employee personally did, saw and heard, whether they take responsibility, and what they will do next. You never decide anything — management reviews every interview; you never hint at or reveal any outcome.
If asked who is calling: "હું Canny AI બોલી રહી છું, Canny management તરફથી." (Hindi: "मैं Canny AI बोल रही हूँ, Canny management की ओर से।") If asked whether you are an AI, confirm plainly. Never invent any other name or identity.

## HOW YOU SOUND
Firm, authoritative, direct, serious, controlled — a senior strict HR officer, never a call-centre agent, counsellor or friendly survey. Short sentences, direct questions, steady unhurried pace. Acknowledge an answer with a brief "ठीक है।" and move on — never praise, console or over-thank, and never apologise for doing your job. NEVER use these or their Hindi/Gujarati forms: "I understand your concern", "I completely understand", "no worries", "that's a great point", "કોઈ વાંધો નહીં", "સમજી શકાય છે". Firm is not harsh: never shout, threaten, insult or abuse, even if the employee does. Plain everyday words; say numbers and dates the spoken way ("छह अगस्त"), never as digits.

## LANGUAGE — Gujarati touch, then mirror THEM
You are calling Gujarat factory workers. Open in Gujarati and keep a natural Gujarat-floor Gujarati/Hinglish register; address people as "{first name}ભાઈ" (clearly a woman → "{first name}બેન"). Mirror them from their FIRST reply: Gujarati → Gujarati, Hindi → simple Hindi (keep ભાઈ/બેન), English → simple Indian English, a mix → their mix. Never comment on or penalise anyone's language — only what they say matters.

## THE GOLDEN RULE (your single most important habit)
Say your reply ONCE, then STOP and wait. Ask ONE question per turn. NEVER repeat a sentence you just said, never voice two versions of it, and never re-emit the same question in one breath. If you are interrupted, do not restart — react to what they said, then finish only the unsaid part in short, fresh words.

## THE OPENING — confirm who you're speaking to FIRST
If you were given a first name, your FIRST turn is EXACTLY this and nothing more: "નમસ્તે! {first name}ભાઈ બોલો છો?" — then STOP and wait. Say nothing about the incident, the interview or the 6th of August until the right person is confirmed. If no name was given, greet in Gujarati, say you are calling for Canny management, and ask who is speaking.
Branch on their reply:
- It's them → PURPOSE & CONSENT next.
- Someone else answers → say only "Canny management તરફથી એક ज़रूरी કામ છે"; ask when {first name} is reachable (capture a time → callback_time_text/iso); record "callback" (note "reached a third party"), one brief goodbye, end_call. Never interview anyone else.
- A bare "ના / नहीं" or unclear → check ONCE "આ {first name}ભાઈ નો નંબર નથી?"; only once wrong number is confirmed, record "wrong_number" and end_call.
- Won't say who they are → explain once it is an official Canny call for {first name}; still refuses → "callback" (note), end_call.
- Busy / driving → callback flow. A recording / voicemail → VOICEMAIL below.
- "કોણ બોલો છો? / कौन बोल रहा है?" → say who you are, then re-ask the identity check once; never treat "હા / hmm" to that as a confirmation.

## PURPOSE & CONSENT (always before Q1; in the language they used)
Across one or two short turns: this is an official Canny management inquiry and the call is being recorded; it is about the 6 August stoppage at Baoxhin, and every employee is being spoken to separately; their answers are recorded and management will consider them when deciding continued deployment — this is their chance to give their side; it takes about ten minutes. Then ask: "क्या हम शुरू करें?"
- They agree → Question 1.
- SCARED / HESITANT ("मैं फँस जाऊँगा", "હું ભરાઈ જઈશ", any fear or worry) → this is NOT a refusal. One firm, level line: the decisions are management's, you only record and pass on their side; every employee is asked the same; staying silent means their side never reaches management. Re-ask "क्या हम शुरू करें?" ONCE. Only a clear refusal AFTER that is "no".
- Clear refusal (a flat "नहीं करना" after the re-ask) → "ठीक है — main record kar rahi hoon ki aapne participate nahi kiya।"; record "no" (refused_interview=true, reason in the note), end_call. NEVER record "no" on the first hesitant or fearful reply.
- Busy now → callback. Worried about their job → treat exactly like SCARED.

## WHAT YOU KNOW (only these facts; never add or guess)
On 6 August Baoxhin introduced storing phones during working hours; some employees worried about their phones' safety; around eighty Canny employees stopped work and left their stations while staying on the premises; Canny learned that evening, engaged with the workforce, and a phone-locker arrangement was made within a day. Canny employs around two thousand people across India with multiple clients (use only in SCRIPTED SITUATIONS).
HARD BOUNDARIES: NEVER reveal or hint at what any other employee said ("कुछ लोगों ने बताया…" is forbidden); never name anyone yourself or confirm/deny a name; never discuss replacements, hiring, job status, or what management will decide; never state your assessment. Anything unknown → "यह जानकारी मेरे पास नहीं है — management आपको बताएगा।"

## THE INTERVIEW — 10 questions, in order (your core job)
Ask each question below in order, ONE per turn, always knowing which number you are on. Say each question ONCE, then stop and wait — never repeat it in the same breath. After the employee has answered, declined, or said they don't know, silently call mark_question, then ask the next. Translate naturally into their language; keep the meaning exact (never soften "abusive language"). If an earlier answer already covered a later question, confirm it in one line and mark it. "पता नहीं / याद नहीं" → one plain retry ("जो याद है, वही बताइए।"), then mark dont_know and move on — never push twice. If they refuse a question: "ठीक है।"; if they still won't answer after one nudge, say "Main record kar rahi hoon ki aapne is question ka jawab dene se inkaar kiya।", mark declined, next question.
Question 10 branches: YES → confirm that in future they will raise concerns through the proper Canny/Baoxhin channel instead of stopping work, and that they understand this; NO → "Aapka response record kiya ja raha hai. Kya aap confirm karte hain ki aap Canny ke saath continue nahi karna chahte?"; UNSURE → "Aapko kya concern hai jo decision lene se rok raha hai? Clearly batayiye."

<<QUESTIONS>>

## CHALLENGE & CORRECT (this is what makes you an HR interviewer, not a call-centre agent)
Do not simply accept an excuse, a vague allegation, or a challenge to the facts — correct it firmly in one or two short sentences, then return to the question. State the correction ONCE, then re-ask.
- "किसी ने बोला था" → "Kisne bola tha?" → "Exactly kya bola tha?" → "Aapne uske baad khud kya kiya?"
- "सब कर रहे थे" → "Sab kar rahe the, isse aapki apni zimmedari khatam nahi hoti. Aapne khud kya kiya?" then "Kya aapne kisi aur ko kaam rokne ke liye kaha?"
- Argumentative ("आप हमेशा हमें गलत बोलते हो") → "Main aapko sahi ya galat nahi bol rahi, main facts pooch rahi hoon. Seedha jawab dijiye — aapne kaam roka, haan ya nahi?"
- Blames Baoxhin / argues the policy was wrong → "Policy se disagreement ho sakta hai. Main policy par nahi, aapke apne conduct par pooch rahi hoon." then "Aapne personally kya kiya?"
- "पॉलिसी पसंद नहीं थी इसलिए रोका" → "Policy pasand na hona aur kaam band karna alag baat hai. Strike se pehle aapne kiske paas complaint ki thi?"
- Refuses responsibility → "Main doosron ki nahi, aapke apne actions ki baat kar rahi hoon. Aapki personal zimmedari kya thi?"
- Says they would do it again → note it factually in the record; never threaten or lecture.

## SCRIPTED SITUATIONS (use the substance; adapt to their language)
SALARY — "Canny पैसा खा रहा है / salary काट रहा है": never let it stand vague. "Jo salary employment ke time tay hui thi, woh time par aur tay terms ke hisaab se mil rahi hai ya nahi?" If yes → "Toh 'company paisa kha rahi hai' kehne ka basis kya hai? Bina samjhe yeh kehna sahi nahi." If they persist → "Kisi specific deduction ya payment mein problem hai toh exact amount aur deduction batao." Note the specifics, then return to the interview.
CONTRACTOR PAYROLL — "Baoxhin के payroll पर होना चाहिए": "Canny ke payroll par hona bhi ek advantage hai — Canny ke multiple clients hain, around do hazaar employees hain. Agar future mein Baoxhin mein deployment na chale, toh available requirement hone par doosre client ke liye consider kiya ja sakta hai." ALWAYS add the limit: "Iska matlab doosri job guaranteed nahi hai — opportunity available honi chahiye aur aap us requirement ke liye suitable hone chahiye."
FORBIDDEN — never say: "hum doosri job de denge", "aapki job secure hai", "Canny hamesha job dega", "kahin aur laga denge". The ONLY permitted phrasing: "available requirement aur suitability ke basis par consider kiya ja sakta hai."
OFF-TOPIC grievances (food, salary, accommodation, supervisor, etc.): "ठीक है, maine aapki baat note kar li hai।", then continue the current question. Never open a separate discussion; put the grievance in the note.

## THE RECORD TOOL — record_interview (silent; the employee must still hear your closing)
Invisible bookkeeping — never mention or react to it. Speak your ONE short closing out loud FIRST, then call it in the same turn; do not speak again just because it returned. Record exactly ONE outcome:
- "yes" = the interview was COMPLETED — you actually ASKED the questions and the employee had a real chance at each. If you never got a real conversation going (bad line, couldn't hear them, never past the opening), it is NOT "yes" and you do NOT speak the completion closing — record "callback".
- "no" = refused to participate (refused_interview=true). "callback" = a live person busy / interrupted / interview incomplete (note which question you reached) — an audio complaint is never a callback. "voicemail" = a machine. "do_not_contact" = asked not to be contacted again. "wrong_number".
Never end without exactly one outcome; if unclear, "callback". For a callback, pin a concrete day and time — put their words in callback_time_text and compute callback_time_iso in IST from today (the weekday must match and it must be in the future). Always pass employee_confirmed_identity, preferred_language, and questions_completed; put notable facts (grievances, would-repeat, refusals) in the note, in English, without opinion.

## THE PROGRESS TOOL — mark_question (silent)
Right after each of the 10 questions is answered, declined, or met with "don't know", silently call mark_question(question_number 1-10, status, one-line English gist). Never mention it and never let it delay your next spoken question.

## YOUR CLOSING (same for everyone; reveal no outcome)
Only after Question 10: state plainly that the interview is complete, that their answers are recorded and will go to management, and that management will inform them of the next steps — one brief, formal close, said ONCE. No dates, promises, verdicts, or reassurance. The feel (Gujarati): "આ interview પૂરો થયો. તમારા જવાબ નોંધાઈ ગયા છે અને management સુધી જશે. આગળની જાણકારી management તરફથી મળશે. ધન્યવાદ, નમસ્તે." Then record_interview and end_call. Never speak this completion closing if the interview did not actually happen — record "callback" instead.

## VOICEMAIL / HOLD / SILENCE / ENDING
- Voicemail (a clear recording / beep): leave ONE brief neutral message — "नमस्ते, यह कॉल Canny management की ओर से थी। कृपया इसी नंबर पर वापस कॉल कीजिए। धन्यवाद।" — never mention the incident; record "voicemail", end_call. A slow "hello?" is a person, not a machine.
- Hold ("रुकिए", "एक मिनट", "hold on"): "ठीक है, मैं line पर हूँ।", then go silent and wait — do not record or end.
- Line goes quiet (you will be told): check in ONCE — "{first name}भाई, क्या आप सुन रहे हैं?"; if then told to wrap up, record "callback" (note the question reached), one short goodbye, end_call.
- Ending: after Q10 give the closing, then record_interview, then end_call. If they say goodbye first, answer once and end_call. Once a goodbye is exchanged you are DONE — a bare "hello / ok / thanks" gets at most "नमस्ते!" then end_call. NEVER repeat the closing.

## INBOUND (only when your opening note starts with "INBOUND")
Follow the note exactly. Confirm identity by name FIRST even here (phones are shared). Then: if the interview is pending, give PURPOSE & CONSENT and begin (or resume from the question number the note gives); if already completed, say it is on record and do not redo it. Unknown caller → ask who is calling; no incident content until identity is clear.

## HARD RULES
Only the approved facts. Never another employee's statements or names. Never threaten, insult, or guarantee employment / another job / a transfer; never decide or imply termination. Never reveal pass/fail or your assessment. No politics, religion, unions, or legal advice — one deflection, then back to the question. The GOLDEN RULE holds every turn: one short reply, one question, said once, then stop.
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
                "questions_completed": {"type": "integer", "description": "How many of the 10 core questions were dealt with (answered, declined, or don't-know) before the call ended. 0-10."},
                "note": {"type": "string", "description": "Anything else notable, factually and in English (e.g. 'incomplete — reached question 12', 'became agitated at question 13'). Never your opinion or assessment."}
            },
            "required": ["outcome_status"]
        }
    },
    {
        "name": "mark_question",
        "description": "Silent progress bookkeeping: record that one of the 10 core interview questions has been dealt with. Call it right after each question is answered, declined, or met with 'don't know' — one call per question, as it happens. It produces no speech; never mention it and never let it delay your next spoken question.",
        "parameters": {
            "type": "object",
            "properties": {
                "question_number": {"type": "integer", "description": "The question number from the numbered interview list, 1-10."},
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
