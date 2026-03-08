# 在一起 — Skills Reference

Implementation-level API details for each capability. Agents reference these when building or modifying features.

---

## Skill: Gemini Live Audio Translation

**Owner:** Backend Agent
**Files:** `server.py` (Room class)

- Model: `gemini-2.5-flash-native-audio-latest`
- Input: PCM 16kHz mono Int16 via `session.send_realtime_input(audio=types.Blob(...))`
- Output: PCM 24kHz mono (audio) + text transcripts
- Session: `gemini_client.aio.live.connect()` → store context manager as `self._session_context`
- System prompt sets translation pair, emotional matching, invisible bridge persona
- `_receive_loop` processes `response.server_content.model_turn.parts` for audio + text

```python
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
        )
    ),
)
```

---

## Skill: Trigger Phrase Detection

**Owner:** Backend Agent
**Files:** `server.py` (`check_trigger()`, `_activate_vision_window()`)

- Runs on every text transcript from Gemini
- Case-insensitive substring match against dictionaries:

```python
VISUAL_TRIGGERS = {
    "en": ["where are you", "show me", "can i see", "what's around you", "let me see"],
    "hi": ["तुम कहाँ हो", "मुझे दिखाओ", "आसपास क्या है", "मैं देख सकता"],
    "zh": ["你在哪", "给我看看", "你那边什么样", "让我看看", "你在哪里"],
}
```

- On match: activate 8s vision window, broadcast `trigger` message to target participant's frontend

---

## Skill: Google Cloud Vision API

**Owner:** Backend Agent + Google API Agent
**Files:** `server.py` (`analyze_frame_with_vision()`)

- Client: `vision.ImageAnnotatorClient()`
- Single batched RPC: `batch_annotate_images()` with `LABEL_DETECTION` (5 results) + `FACE_DETECTION` (1 result)
- Key moment filter: only broadcast when labels differ >50% from previous + 8s debounce
- Face emotions: `joy_likelihood`, `sorrow_likelihood`, `anger_likelihood`, `surprise_likelihood` — broadcast when score >= 4 (LIKELY)
- Cost: ~$0.0015/image

---

## Skill: Gemini Vision Narration

**Owner:** Backend Agent
**Files:** `server.py` (`narrate_frame_with_gemini()`)

- Model: `gemini-2.5-flash`
- Throttled to 1 per 10 seconds
- Prompt: "Describe the scene to the listener in {target_lang}. Be brief (1-2 sentences), warm, and natural."
- Output sent as `narration` message type to frontend

---

## Skill: Frontend Audio Capture & Playback

**Owner:** Frontend Agent
**Files:** `static/participant.html`

- Capture: `getUserMedia({audio: {sampleRate: 16000, channelCount: 1, echoCancellation: true}})` → `AudioContext(16kHz)` → `ScriptProcessor(4096)` → Float32→Int16 → `ws.send(buffer)`
- Playback: queue-based drain, rate-matched `AudioContext` (16kHz or 24kHz)
- Reuse AudioContext instances (`ctx16k`, `ctx24k`) — never create per chunk

---

## Skill: Camera + Motion Detection

**Owner:** Frontend Agent
**Files:** `static/participant.html`

- Hold-to-activate camera via pointer events on cam button
- Frame capture: 320x240 JPEG at 3s intervals → server as `video_frame` message
- Motion overlay: 80x60 downsampled, 10px block differencing, green arrows at centroids of changed regions (threshold 28, 300ms interval)
- Canvas overlay on `.cam-pip` container

---

## Skill: Interactive Storybook (Gemini Interleaved Output)

**Owner:** Google API Agent
**Files:** `storybook_generator.py`

- Model: `gemini-2.5-flash-image`
- Config: `response_modalities=[Modality.TEXT, Modality.IMAGE]`
- Input: screenshots as `inline_data` (JPEG base64) + text prompt with voice snippets + scene descriptions
- Output: interleaved `part.text` and `part.inline_data` → storybook pages
- Rendered as standalone HTML via `render_storybook_html()`
- Handles screenshots as dict, str, or bytes via `isinstance()` checks

---

## Skill: Image Stylization

**Owner:** Google API Agent
**Files:** `memory_video.py` (`stylize_image()`, `stylize_all_images()`)

- Model: `gemini-2.5-flash-image`
- Input: base64 JPEG screenshot + artistic style prompt
- Output: bytes (image/png) — warm, dreamlike, watercolor aesthetic
- Concurrent stylization via `asyncio.gather()`
- Cost: ~$0.02/image

---

## Skill: Memory Video (Veo)

**Owner:** Google API Agent
**Files:** `memory_video.py` (`generate_memory_video()`)

- Model: `veo-2.0-generate-001`
- Input: first stylized image (bytes) + narrative prompt
- Config: 1 video, 8 seconds, `person_generation="allow_all"`
- Polling: every 5s, max 300s, with retry on transient errors
- Output: MP4 bytes
- Cost: ~$2.40 per generation

---

## Skill: ElevenLabs Fallback Pipeline

**Owner:** Backend Agent
**Files:** `server.py` (`_handle_fallback_audio()`, `elevenlabs_stt()`, `elevenlabs_tts()`)

- STT: ElevenLabs Scribe v2 (~100ms latency)
- Translation: Gemini `gemini-2.5-flash-lite` text-to-text
- TTS: ElevenLabs Flash v2.5 (~75ms latency)
- Triggered when `room.use_fallback = True`
- Audio routing unchanged — same WebSocket path

---

## Skill: Billing Monitor

**Owner:** Orchestrator Agent
**Files:** `billing_monitor.py`, `server.py` (`log_api_call()`)

- Tracks all API calls with estimated costs
- Warning thresholds: $3, $5, $9, $12, $15, $19, $21, $23, $24.90
- REST endpoint: `GET /api/billing`
- Frontend polls every 15 seconds
- Console warnings logged at each threshold breach
