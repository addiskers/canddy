"""plivo_handler.py — codec tables, goodbye/mute separation, sender resilience, squelch."""

import asyncio
import base64
import json
import struct
import time

import pytest

import plivo_handler as ph
from plivo_handler import (PlivoMediaBridge, _has_closing_repeat, _looks_like_agent_question,
                           _looks_like_goodbye,
                           _MULAW_DECODE, _MULAW_SQ, _PCM_TO_ULAW, _SILENCE_20MS_16K,
                           _mulaw_frame_meansquare, _pcm16_to_mulaw_sample, pcm24k_to_mulaw)

try:
    from starlette.websockets import WebSocketState
except ImportError:
    WebSocketState = None


class FakeWS:
    def __init__(self, fail_times=0, connected=True, incoming=None):
        self.sent = []
        self.fail_times = fail_times
        self._incoming = list(incoming or [])
        if WebSocketState is not None:
            state = WebSocketState.CONNECTED if connected else WebSocketState.DISCONNECTED
            self.client_state = state
            self.application_state = state

    async def send_json(self, payload):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("boom")
        self.sent.append(payload)

    async def receive_text(self):
        if not self._incoming:
            raise Exception((1000,))
        return self._incoming.pop(0)


def _bridge(ws=None, **env):
    return PlivoMediaBridge(ws or FakeWS(), gemini_client=None, text_trigger="[go]")


# Codec tables

def test_pcm_to_ulaw_table_matches_reference_encoder():
    for s in (-32768, -32635, -10000, -1, 0, 1, 42, 8000, 32635, 32767):
        assert _PCM_TO_ULAW[s & 0xFFFF] == _pcm16_to_mulaw_sample(s)


def test_mulaw_square_table_matches_decode_table():
    for b in range(256):
        assert _MULAW_SQ[b] == int(_MULAW_DECODE[b]) ** 2


def test_pcm24k_to_mulaw_lookup_equivalent_to_per_sample_encode():
    samples = [0, 100, -100, 8000, -8000, 32767, -32768, 5, 6, 7]
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    expected = bytes(_pcm16_to_mulaw_sample((samples[i] + samples[i+1] + samples[i+2]) // 3)
                     for i in range(0, len(samples) - 2, 3))
    assert pcm24k_to_mulaw(pcm) == expected


def test_meansquare_zero_for_silence_high_for_speech():
    silence = bytes([_pcm16_to_mulaw_sample(0)]) * 160
    loud = bytes([_pcm16_to_mulaw_sample(8000)]) * 160
    assert _mulaw_frame_meansquare(silence) < 10
    assert _mulaw_frame_meansquare(loud) > 250_000


# Goodbye heuristics (Hindi-first interview: matchers must handle native script)

def test_looks_like_goodbye():
    assert _looks_like_goodbye("okay bye") is True
    assert _looks_like_goodbye("thank you") is True
    assert _looks_like_goodbye("धन्यवाद, रखता हूँ") is True                 # Hindi sign-off
    assert _looks_like_goodbye("ठीक है") is False       # conservative — common mid-interview ack
    assert _looks_like_goodbye("bye, but what time is it?") is False       # real follow-up
    assert _looks_like_goodbye("thanks a lot for calling me today about this event") is False  # too long
    assert _looks_like_goodbye("") is False


def test_has_closing_repeat_detects_doubled_closing():
    assert _has_closing_repeat("aapke jawab note ho gaye hain dhanyavaad "
                               "aapke jawab note ho gaye hain dhanyavaad") is True   # 5-gram repeat
    assert _has_closing_repeat("thank you for your time, goodbye") is False
    # Doubled Devanagari closing in ONE turn — pins the Unicode-aware normalisation:
    # the old [^a-z0-9 ] scrub stripped ALL Devanagari and left this guard dead on Hindi calls.
    assert _has_closing_repeat(
        "आपके जवाब दर्ज हो गए हैं और आगे की जानकारी management देगा ... "
        "आपके जवाब दर्ज हो गए हैं और आगे की जानकारी management देगा"
    ) is True
    # A single Hindi closing must NOT read as a repeat
    assert _has_closing_repeat("आपके जवाब दर्ज हो गए हैं, धन्यवाद") is False
    # Real failure mode kept from production: paraphrased double-apology/intro in one turn
    doubled = (
        "Oh, sorry about that! Let me try again. I am calling on behalf of Canny "
        "management about an official matter. Oh, sorry about that! Yes, I was "
        "calling on behalf of Canny management."
    )
    assert _has_closing_repeat(doubled) is True
    # English closing marker voiced twice ("thank you for your time")
    assert _has_closing_repeat(
        "thank you for your time. your answers will go to management. "
        "thank you for your time, goodbye."
    ) is True


def test_looks_like_agent_question():
    assert _looks_like_agent_question("क्या आप सुन रहे हैं?") is True       # Hindi still-there check
    assert _looks_like_agent_question("Could you explain what happened after that?") is True
    assert _looks_like_agent_question("Are you still there?") is True
    assert _looks_like_agent_question("Oh wonderful, so glad you'll be there!") is False
    assert _looks_like_agent_question("") is False


def test_repeat_suppress_flushes_queued_playout():
    """When a doubled closing is detected mid-turn, already-queued frames must be drained
    and Plivo clearAudio sent — otherwise the second closing still plays."""
    async def run():
        ws = FakeWS()
        b = _bridge(ws)
        b.stream_id = "s1"
        b._agent_audio_started = True
        # Pretend a full closing is already queued for playout
        for _ in range(5):
            await b._out_frames.put(b"\xff" * 160)
        b._residual.extend(b"\xaa" * 40)
        # Drive the gemini-event path that arms suppress + flush
        b._turn_text = ""
        # Simulate accumulating a doubled Hindi closing via gemini text events
        event = {"type": "gemini", "text": (
            "आपके जवाब दर्ज हो गए हैं और आगे की जानकारी management देगा। "
            "आपके समय के लिए धन्यवाद। आपके जवाब दर्ज हो गए हैं और आगे की "
            "जानकारी management देगा। आपके समय के लिए धन्यवाद।"
        )}
        # Inline the same logic _gemini_loop uses for gemini events
        b._turn_open = True
        b._turn_text += " " + event["text"]
        assert _has_closing_repeat(b._turn_text) is True
        b._suppress_turn = True
        await b._flush_playout()
        assert b._out_frames.empty()
        assert not b._residual
        assert any(p.get("event") == "clearAudio" for p in ws.sent)
        # Further audio while suppressed is dropped
        await b.audio_output_callback(b"\x00\x10" * 240)
        assert b._out_frames.empty() and not b._residual
        return True
    assert asyncio.run(run()) is True


def test_post_record_closing_schedules_muted_hangup():
    """After record_interview + a spoken closing turn_complete, hang up muted so bare Hello can't re-engage."""
    async def run():
        b = _bridge()
        b.stream_id = "s1"
        b._agent_audio_started = True
        b._rsvp_recorded = True
        b._post_rsvp_hangup_armed = True
        b._spoke_since_user = True
        b._last_agent_audio = time.monotonic()
        b._turn_text = "आपके जवाब दर्ज हो गए हैं। आपके समय के लिए धन्यवाद।"
        # Mimic turn_complete branch
        b._last_agent_asked_question = _looks_like_agent_question(b._turn_text)
        assert b._last_agent_asked_question is False
        if (b._post_rsvp_hangup_armed and not b._post_rsvp_closing_done
                and b._spoke_since_user
                and not (b._pending_hangup_task and not b._pending_hangup_task.done())):
            b._post_rsvp_closing_done = True
            b._post_rsvp_hangup_armed = False
            b._schedule_end(mute=True)
        assert b._ending is True
        assert b._pending_hangup_task is not None
        b._pending_hangup_task.cancel()
        return True
    assert asyncio.run(run()) is True


# Goodbye playback: scheduling a hangup must not mute the farewell

def test_schedule_end_soft_lets_farewell_audio_through():
    async def run():
        b = _bridge()
        b.stream_id = "s1"
        b._agent_audio_started = True
        b._schedule_end(mute=False)
        assert b._ending is False                     # farewell may still play
        await b.audio_output_callback(b"\x00\x10" * 240)   # 24kHz pcm chunk
        got = not b._out_frames.empty() or bool(b._residual)
        b._pending_hangup_task.cancel()
        return got
    assert asyncio.run(run()) is True


def test_schedule_end_muted_drops_further_audio():
    async def run():
        b = _bridge()
        b.stream_id = "s1"
        b._agent_audio_started = True
        b._schedule_end(mute=True)
        assert b._ending is True
        await b.audio_output_callback(b"\x00\x10" * 240)
        got = b._out_frames.empty() and not b._residual
        b._pending_hangup_task.cancel()
        return got
    assert asyncio.run(run()) is True


# Outbound sender resilience

def test_sender_survives_transient_send_failures():
    async def run():
        ws = FakeWS(fail_times=3, connected=True)
        b = _bridge(ws)
        b.stream_id = "s1"
        for _ in range(5):
            b._out_frames.put_nowait(b"\x00" * 160)
        task = asyncio.create_task(b._outbound_sender())
        await asyncio.sleep(0.3)
        alive = not task.done()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        # 3 transient failures skipped, remaining 2 frames actually sent
        return alive, len(ws.sent)
    alive, sent = asyncio.run(run())
    assert alive is True
    assert sent == 2


@pytest.mark.skipif(WebSocketState is None, reason="starlette not installed")
def test_sender_exits_when_socket_is_closed():
    async def run():
        ws = FakeWS(fail_times=99, connected=False)
        b = _bridge(ws)
        b.stream_id = "s1"
        b._out_frames.put_nowait(b"\x00" * 160)
        task = asyncio.create_task(b._outbound_sender())
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
            return False
        return True
    assert asyncio.run(run()) is True


# Noise squelch: substitutes silence, same frame cadence, never drops

def _media_msg(mulaw: bytes) -> str:
    return json.dumps({"event": "media",
                       "media": {"payload": base64.b64encode(mulaw).decode(), "track": "inbound"}})


def test_squelch_substitutes_silence_and_preserves_frame_count(monkeypatch):
    monkeypatch.setenv("EO_NOISE_GATE", "true")
    quiet = bytes([_pcm16_to_mulaw_sample(0)]) * 160
    loud = bytes([_pcm16_to_mulaw_sample(8000)]) * 160

    async def run():
        ws = FakeWS(incoming=[_media_msg(quiet), _media_msg(quiet), _media_msg(loud)])
        b = _bridge(ws)
        b._rec_on = False
        await b.handle_plivo_messages()
        frames = []
        while not b.audio_input_queue.empty():
            frames.append(b.audio_input_queue.get_nowait())
        return frames

    frames = asyncio.run(run())
    assert len(frames) == 3                          # cadence preserved — nothing dropped
    assert frames[0] == _SILENCE_20MS_16K            # below gate, no recent voice → silence
    assert frames[1] == _SILENCE_20MS_16K
    assert frames[2] != _SILENCE_20MS_16K            # voiced frame passes unmodified


def test_gate_off_forwards_everything_verbatim(monkeypatch):
    monkeypatch.setenv("EO_NOISE_GATE", "false")
    quiet = bytes([_pcm16_to_mulaw_sample(0)]) * 160

    async def run():
        ws = FakeWS(incoming=[_media_msg(quiet)])
        b = _bridge(ws)
        b._rec_on = False
        await b.handle_plivo_messages()
        return b.audio_input_queue.get_nowait()

    frame = asyncio.run(run())
    assert len(frame) == 640
    # decoded silence upsampled is all-zero PCM, but it went through the codec path
    assert frame == ph.mulaw_to_pcm16k(quiet)


# Silence check: ask once, then escalate — never loop the question

def test_silence_nudge_fires_once_then_escalates(monkeypatch):
    monkeypatch.setenv("EO_SILENCE_CHECK", "true")
    monkeypatch.setenv("EO_SILENCE_PROMPT_SECONDS", "0.2")
    monkeypatch.setenv("EO_SILENCE_HANGUP_SECONDS", "0.5")

    async def run():
        b = _bridge()
        b.stream_id = "s1"
        b._agent_audio_started = True
        b.first_name = "Pratik"
        t = time.monotonic()
        b._last_agent_audio = t - 10          # agent finished long ago, caller silent since
        b._last_caller_audio = t - 10
        b._last_activity = t - 10
        task = asyncio.create_task(b._idle_hangup_guard())
        await asyncio.sleep(1.1)              # 1st guard tick → "are you still there?"
        await b.audio_output_callback(b"\x00\x10" * 240)   # the nudge is spoken aloud...
        b._drain_outbound()                                # ...and finishes playing
        await asyncio.sleep(2.2)              # must ESCALATE now, not re-ask
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if b._pending_hangup_task:
            b._pending_hangup_task.cancel()
        msgs = []
        while not b.text_input_queue.empty():
            msgs.append(b.text_input_queue.get_nowait())
        return msgs

    msgs = asyncio.run(run())
    still_there = [m for m in msgs if "सुन रहे हैं" in m]     # Hindi are-you-still-there nudge
    wrapups = [m for m in msgs if "seems dead" in m]
    assert len(still_there) == 1, f"nudge must fire exactly once, got {msgs}"
    assert len(wrapups) == 1, f"expected one wrap-up escalation, got {msgs}"
    assert "Pratik" in still_there[0]


# Greeting watchdog: one firm push when Gemini stalls on the opening line

def test_greeting_watchdog_pushes_exactly_once(monkeypatch):
    monkeypatch.setenv("EO_GREETING_NUDGE_SECONDS", "0.5")

    async def run():
        b = _bridge()
        b._greeting_sent_at = time.monotonic() - 5    # trigger sent, still no agent audio
        task = asyncio.create_task(b._idle_hangup_guard())
        await asyncio.sleep(2.3)                      # two+ guard ticks
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        msgs = []
        while not b.text_input_queue.empty():
            msgs.append(b.text_input_queue.get_nowait())
        return msgs

    msgs = asyncio.run(run())
    assert sum("Speak your opening line" in m for m in msgs) == 1


def test_greeting_watchdog_never_fires_after_audio_started(monkeypatch):
    monkeypatch.setenv("EO_GREETING_NUDGE_SECONDS", "0.5")

    async def run():
        b = _bridge()
        b._greeting_sent_at = time.monotonic() - 5
        b._agent_audio_started = True                 # greeting already played
        task = asyncio.create_task(b._idle_hangup_guard())
        await asyncio.sleep(1.2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        msgs = []
        while not b.text_input_queue.empty():
            msgs.append(b.text_input_queue.get_nowait())
        return [m for m in msgs if "Speak your opening line" in m]

    assert asyncio.run(run()) == []


def test_greeting_rescue_fires_once_for_noise_killed_opening():
    async def run():
        b = _bridge()
        b._greeting_sent_at = time.monotonic() - 1    # opening queued, zero caller evidence yet
        await b._maybe_rescue_greeting()
        await b._maybe_rescue_greeting()              # a second interrupt: once-per-call guard holds
        msgs = []
        while not b.text_input_queue.empty():
            msgs.append(b.text_input_queue.get_nowait())
        return msgs

    msgs = asyncio.run(run())
    assert sum("Say your opening line again" in m for m in msgs) == 1


def test_greeting_rescue_stays_quiet_for_real_speech_or_after_a_turn():
    async def run():
        results = []
        for field, value in (("_last_user_event", time.monotonic()),
                             ("_last_caller_audio", time.monotonic()),
                             ("_any_turn_complete", True)):
            b = _bridge()
            b._greeting_sent_at = time.monotonic() - 1
            setattr(b, field, value)
            await b._maybe_rescue_greeting()
            results.append(b.text_input_queue.empty())
        return results

    assert asyncio.run(run()) == [True, True, True]


def test_missed_reply_rescue_fires_when_caller_speech_goes_unanswered(monkeypatch):
    """Caller spoke AFTER the agent's last audio and got nothing for 4s → prompt the
    agent once. The still-there ladder can't cover this (it measures silence, and a
    talking caller keeps resetting it)."""
    monkeypatch.setenv("EO_SILENCE_CHECK", "true")
    monkeypatch.setenv("EO_UNANSWERED_REPLY_SECONDS", "0.3")

    async def run():
        b = _bridge()
        b._agent_audio_started = True
        t = time.monotonic()
        b._last_agent_audio = t - 10              # greeting ended long ago
        b._last_caller_audio = t - 5              # caller replied 5s ago...
        b._last_activity = t - 5                  # ...and nothing since
        task = asyncio.create_task(b._idle_hangup_guard())
        await asyncio.sleep(2.3)                  # two+ ticks: must fire exactly once
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if b._pending_hangup_task:
            b._pending_hangup_task.cancel()
        msgs = []
        while not b.text_input_queue.empty():
            msgs.append(b.text_input_queue.get_nowait())
        return [m for m in msgs if "said something" in m]

    assert len(asyncio.run(run())) == 1


def test_missed_reply_rescue_fires_while_caller_keeps_talking(monkeypatch):
    """THE shivi case: caller repeats 'hello hello' every 2s (fresh voiced frames) while
    the agent stays mute. The rescue must key on the AGENT's silence — the caller's
    repeats must not keep resetting it."""
    monkeypatch.setenv("EO_SILENCE_CHECK", "true")
    monkeypatch.setenv("EO_UNANSWERED_REPLY_SECONDS", "0.3")

    async def run():
        b = _bridge()
        b._agent_audio_started = True
        t = time.monotonic()
        b._last_agent_audio = t - 10              # agent mute for 10s...
        b._last_caller_audio = t - 1              # ...caller spoke just 1s ago (still trying)
        b._last_activity = t - 1
        task = asyncio.create_task(b._idle_hangup_guard())
        await asyncio.sleep(1.5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if b._pending_hangup_task:
            b._pending_hangup_task.cancel()
        msgs = []
        while not b.text_input_queue.empty():
            msgs.append(b.text_input_queue.get_nowait())
        return [m for m in msgs if "said something" in m]

    assert len(asyncio.run(run())) == 1


def test_missed_reply_rescue_stays_quiet_when_agent_already_replied(monkeypatch):
    monkeypatch.setenv("EO_SILENCE_CHECK", "true")
    monkeypatch.setenv("EO_UNANSWERED_REPLY_SECONDS", "0.3")

    async def run():
        b = _bridge()
        b._agent_audio_started = True
        t = time.monotonic()
        b._last_caller_audio = t - 5
        b._last_agent_audio = t - 2               # agent DID reply after the caller
        b._last_activity = t - 2
        task = asyncio.create_task(b._idle_hangup_guard())
        await asyncio.sleep(1.2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if b._pending_hangup_task:
            b._pending_hangup_task.cancel()
        msgs = []
        while not b.text_input_queue.empty():
            msgs.append(b.text_input_queue.get_nowait())
        return [m for m in msgs if "said something" in m]

    assert asyncio.run(run()) == []


def test_silence_nudge_fires_even_when_turn_open_flag_is_stuck(monkeypatch):
    """Gemini may never send turn_complete for a text-triggered greeting when the
    caller's speech doesn't register as a turn — the stuck _turn_open flag must not
    muzzle the 'are you still there?' ladder."""
    monkeypatch.setenv("EO_SILENCE_CHECK", "true")
    monkeypatch.setenv("EO_SILENCE_PROMPT_SECONDS", "0.2")

    async def run():
        b = _bridge()
        b._agent_audio_started = True
        b._turn_open = True                       # stuck: turn_complete never arrived
        t = time.monotonic()
        b._last_agent_audio = t - 10              # ...but no agent audio for 10s
        b._last_caller_audio = t - 10
        b._last_activity = t - 10
        task = asyncio.create_task(b._idle_hangup_guard())
        await asyncio.sleep(1.5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if b._pending_hangup_task:
            b._pending_hangup_task.cancel()
        msgs = []
        while not b.text_input_queue.empty():
            msgs.append(b.text_input_queue.get_nowait())
        return [m for m in msgs if "सुन रहे हैं" in m]

    assert len(asyncio.run(run())) == 1


# Silence nudge: cooldown + hard cap survive noise-blip flag resets

def test_silence_nudge_respects_cooldown_after_noise_reset(monkeypatch):
    monkeypatch.setenv("EO_SILENCE_CHECK", "true")
    monkeypatch.setenv("EO_SILENCE_PROMPT_SECONDS", "0.2")
    monkeypatch.setenv("EO_SILENCE_NUDGE_COOLDOWN_S", "60")

    async def run():
        b = _bridge()
        b._agent_audio_started = True
        t = time.monotonic()
        b._last_agent_audio = t - 10
        b._last_caller_audio = t - 10
        b._last_activity = t - 10
        # a nudge fired 1s ago; then a noise blip reset the flag
        b._silence_nudged = False
        b._silence_nudge_at = t - 1
        b._silence_nudge_count = 1
        task = asyncio.create_task(b._idle_hangup_guard())
        await asyncio.sleep(1.5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if b._pending_hangup_task:
            b._pending_hangup_task.cancel()
        msgs = []
        while not b.text_input_queue.empty():
            msgs.append(b.text_input_queue.get_nowait())
        return [m for m in msgs if "सुन रहे हैं" in m]

    assert asyncio.run(run()) == []                   # cooldown blocks the re-ask


# Post-record arming: ONLY the outcome tool arms end-of-call logic

class _FakeGemini:
    """Yields a scripted event stream through the real _gemini_loop plumbing."""

    def __init__(self, events):
        self._events = list(events)

    async def start_session(self, **_kw):
        for e in self._events:
            yield e


def test_mark_question_tool_call_never_arms_post_record_hangup():
    async def run():
        b = PlivoMediaBridge(FakeWS(), gemini_client=_FakeGemini([
            {"type": "tool_call", "name": "mark_question",
             "args": {"question_number": 3, "status": "answered"}, "result": {"success": True}},
        ]), text_trigger="[go]")
        await b._gemini_loop()
        return b._rsvp_recorded, b._post_rsvp_hangup_armed
    recorded, armed = asyncio.run(run())
    assert recorded is False
    assert armed is False


def test_record_interview_tool_call_arms_post_record_hangup():
    async def run():
        b = PlivoMediaBridge(FakeWS(), gemini_client=_FakeGemini([
            {"type": "tool_call", "name": "mark_question",
             "args": {"question_number": 10, "status": "answered"}, "result": {"success": True}},
            {"type": "tool_call", "name": "record_interview",
             "args": {"outcome_status": "yes"}, "result": {"outcome_status": "yes"}},
        ]), text_trigger="[go]")
        await b._gemini_loop()
        return b._rsvp_recorded, b._post_rsvp_hangup_armed
    recorded, armed = asyncio.run(run())
    assert recorded is True
    assert armed is True


def test_tool_sets_are_pinned():
    """The bridge keys end-of-call logic on tool NAMES — pin them at the source."""
    import gemini_live
    assert {t["name"] for t in gemini_live.TOOLS} == {"record_interview", "mark_question", "end_call"}
    assert gemini_live._ASYNC_TOOLS == {"record_interview", "mark_question"}
    # the outcome enum the whole pipeline (validator/labels/CSV) relies on
    tool = next(t for t in gemini_live.TOOLS if t["name"] == "record_interview")
    assert tool["parameters"]["properties"]["outcome_status"]["enum"] == [
        "yes", "no", "callback", "voicemail", "do_not_contact", "wrong_number"]


# Bounded input queue: drop-oldest, never blocks

def test_put_audio_drops_oldest_on_overflow():
    async def run():
        b = _bridge()
        b.audio_input_queue = asyncio.Queue(maxsize=3)
        for i in range(5):
            b._put_audio(bytes([i]) * 4)
        out = []
        while not b.audio_input_queue.empty():
            out.append(b.audio_input_queue.get_nowait())
        return out
    out = asyncio.run(run())
    assert len(out) == 3
    assert out[0][0] == 2 and out[-1][0] == 4        # oldest (0,1) dropped
