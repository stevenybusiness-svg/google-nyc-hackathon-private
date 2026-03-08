# 在一起 — AI Coding Agent Guide

This file defines the four AI coding agents that operate on this codebase. Each agent owns a domain, knows which files to touch, and follows strict conventions. Reference `.cursor/rules/` for file-specific rules and `skills.md` for implementation-level API details.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────┐
│                 Orchestrator Agent                │
│  Coordinates cross-cutting concerns, deployment  │
└────────┬──────────────┬──────────────┬───────────┘
         │              │              │
    ┌────▼────┐   ┌─────▼─────┐  ┌────▼─────┐
    │ Backend │   │ Frontend  │  │ Google   │
    │ Agent   │   │ Agent     │  │ API Agent│
    └─────────┘   └───────────┘  └──────────┘
```

---

## Agent 1: Backend Agent

**Domain:** `server.py`, `billing_monitor.py`, `requirements.txt`, `Dockerfile`

### Responsibilities

- WebSocket server (FastAPI): room management, participant lifecycle, audio routing
- Gemini Live API session management (`_session_context` pattern for async CM lifecycle)
- Audio pipeline: mic PCM → Gemini → translated audio + transcript → correct participant
- Caption routing: `side: "mine"` to speaker, `side: "theirs"` to listener
- Vision pipeline orchestration: trigger detection → frame capture → Vision API + narration
- Screenshot/snippet/scene archival for post-call memorabilia
- ElevenLabs fallback handler (STT → Gemini text translate → TTS)
- Billing monitor integration

### Key Patterns

- **Async context manager lifecycle:** `self._session_context` must be stored on the Room instance to prevent garbage collection. Never use a local variable for the Gemini Live connect context.
- **Audio batching:** Log `gemini_live_audio_input_min` every ~30s (960K bytes), not per chunk.
- **Vision gating:** `_vision_allowed()` enforces trigger windows + 1fps debounce. Only broadcast `visual_caption` when scene labels change >50% from previous.
- **Room archival:** `_archive_room()` preserves screenshots, captions, snippets, scenes for memorabilia generation after all participants leave.

### File Conventions

- All routes use `/api/rooms/{room_id}/...` prefix
- WebSocket endpoint: `/ws/{room_id}/{participant_id}?language=...`
- Static files mounted last: `app.mount("/static", ...)`
- Logging via `logging.getLogger(__name__)`

---

## Agent 2: Frontend Agent

**Domain:** `static/index.html`, `static/participant.html`, `static/memories.html`

### Responsibilities

- Landing page (`index.html`): room creation/join, billing display
- Call UI (`participant.html`): dual-panel book-themed transcript, camera PiP with motion overlay, controls
- Memories page (`memories.html`): post-call storybook, memory video, caption history
- WebSocket client: connect, send audio, receive messages, handle reconnection
- Mic capture: `getUserMedia` → `AudioContext` → `ScriptProcessor` → PCM 16kHz Int16
- Audio playback: queue-based drain with rate-matched `AudioContext` (16kHz/24kHz)
- Camera: hold-to-activate, frame capture at 3s intervals, motion detection at 300ms intervals

### Key Patterns

- **Book layout:** Left page = speaker's own words (`side: "mine"`), right page = other person's translation (`side: "theirs"`). Visual moments (scene/sentiment) go in right panel with `.visual-moment` class.
- **Motion detection:** Canvas overlay on camera PiP. 80x60 downsampled frames, 10px block differencing, green arrows drawn at motion centroids. `MOTION_THRESHOLD = 28`.
- **Message routing:** `handleMessage()` switch on `msg.type`: audio, caption, visual_caption, narration, sentiment, trigger, participant_joined/left, room_state, error.
- **Debouncing:** Visual caption entries: 10s debounce + text dedup. Sentiment entries: 12s debounce.
- **Billing polling:** 15s interval (not 2s) to reduce API load.

### File Conventions

- Inline `<style>` and `<script>` (no external CSS/JS)
- Fonts: Fraunces (serif/headings), DM Sans (UI), Noto Sans SC/Devanagari (CJK/Hindi)
- Color palette: dark background (#0e0e12), warm amber accents (rgba(191,155,120,...))
- No frameworks — vanilla HTML/CSS/JS only

---

## Agent 3: Google API Agent

**Domain:** `memory_video.py`, `storybook_generator.py`, API integration in `server.py`

### Responsibilities

- Gemini Live API: real-time audio translation (`gemini-2.5-flash-native-audio-latest`)
- Gemini text generation: environment narration, text translation fallback (`gemini-2.5-flash`, `gemini-2.5-flash-lite`)
- Gemini image generation: storybook interleaved output, image stylization (`gemini-2.5-flash-image`)
- Google Cloud Vision API: label detection, face/emotion detection (batched single RPC)
- Veo video generation: stylized images → memory video (`veo-2.0-generate-001`)
- ElevenLabs: Scribe v2 STT, Flash v2.5 TTS (fallback pipeline)

### Key Patterns

- **Screenshot data flexibility:** Always handle screenshots as either `dict` with `"data"` key, raw base64 `str`, or `bytes`. Use `isinstance()` checks everywhere.
- **Image data normalization:** Gemini returns `part.inline_data.data` as bytes. Veo needs bytes for `types.Image(image_bytes=...)`. Storybook HTML renderer needs base64 strings. Always convert at the boundary.
- **Veo polling:** `generate_videos()` returns an operation. Poll with `operations.get()` every 5s, max 300s timeout. Handle polling errors gracefully (retry after 3s).
- **Cost awareness:** Every API call goes through `log_api_call()` with estimated cost. See `billing_monitor.py` for rates.

### Model Reference

| Use Case | Model | Notes |
|----------|-------|-------|
| Live translation | `gemini-2.5-flash-native-audio-latest` | Audio-in, audio+text-out |
| Environment narration | `gemini-2.5-flash` | Text generation from image |
| Text translation fallback | `gemini-2.5-flash-lite` | Cheapest for plain translation |
| Storybook + stylization | `gemini-2.5-flash-image` | Interleaved TEXT+IMAGE output |
| Memory video | `veo-2.0-generate-001` | 8s video from single image |

### API Cost Estimates

| Operation | Cost |
|-----------|------|
| Audio input (1 min) | $0.0375 |
| Audio output (1 min) | $0.15 |
| Vision API (1 image) | $0.0015 |
| Storybook (5 screenshots) | ~$0.05 |
| Memory video (8s) | ~$2.40 |

---

## Agent 4: Orchestrator Agent

**Domain:** All files — coordinates cross-cutting concerns

### Responsibilities

- End-to-end integration testing across agents
- Cloud Run deployment (`gcloud run deploy`)
- Environment configuration (`.env`, API keys, quota projects)
- Dependency management (`requirements.txt`, `.venv`)
- Performance optimization: latency, API cost, WebSocket buffering
- Demo readiness: pre-recorded fallbacks, README, presentation flow

### Coordination Rules

1. **Backend ↔ Frontend contract:** WebSocket messages use `type` field. New message types must be documented in both agents' sections above.
2. **Backend ↔ Google API contract:** Room stores screenshots as `list[dict]` with `{"data": base64_str, "timestamp": str, ...}`. Pipelines must accept this format.
3. **Cost budget:** Total demo budget is $25. Every new API call path must include a `log_api_call()` with realistic cost estimate.
4. **Error isolation:** API failures must not crash the WebSocket connection. Use `asyncio.create_task()` for non-critical pipelines (vision, narration). Wrap in try/except.

### Deployment Checklist

- [ ] `requirements.txt` has all deps with versions
- [ ] `.env.example` documents all required keys
- [ ] `python -c "import server"` succeeds
- [ ] `curl localhost:8000/health` returns `{"status": "ok"}`
- [ ] WebSocket connects and stays alive for >60s
- [ ] Two participants can join same room
- [ ] Audio translation produces captions within 3s
- [ ] Camera trigger → vision labels within 5s

---

## WebSocket Message Types (Shared Contract)

| Type | Direction | Fields | Purpose |
|------|-----------|--------|---------|
| `audio` | server→client | `data`, `mime_type` | Translated audio PCM |
| `caption` | server→client | `text`, `speaker`, `side` | Transcript entry |
| `visual_caption` | server→client | `text` | Vision API scene labels (key moments) |
| `narration` | server→client | `text` | Gemini environment narration |
| `sentiment` | server→client | `emotion`, `score` | Face emotion detection |
| `trigger` | server→client | `trigger`, `text`, `target_participant_id` | Activate camera |
| `active_speaker` | server→client | `participant_id` | Who is speaking |
| `barge_in` | server→client | `by` | Interruption notification |
| `room_state` | server→client | `participants`, `languages`, etc. | Room sync |
| `participant_joined` | server→client | `participant_id` | Join notification |
| `participant_left` | server→client | `participant_id` | Leave notification |
| `video_frame` | client→server | `data` (base64 JPEG) | Camera frame |
| `ping`/`pong` | bidirectional | — | Keepalive |
| `error` | server→client | `message` | Error notification |

---

## Parallelism & Concurrency Model

### Principle: Parallel execution, no conflicts

Everything runs in a single asyncio event loop (single Cloud Run worker for WebSocket affinity). Parallelism is achieved through concurrent coroutines, not threads or processes.

### Concurrency Primitives

| Primitive | Scope | Purpose |
|-----------|-------|---------|
| `_gemini_sem` (Semaphore=4) | Global | Caps concurrent Gemini API calls to prevent rate limiting |
| `_vision_sem` (Semaphore=3) | Global | Caps concurrent Vision gRPC calls |
| `room._state_lock` (Lock) | Per-room | Protects `screenshots`, `captions`, `scene_descriptions`, `voice_snippets` |
| `_generation_tasks` (dict) | Global | Tracks background generation task status |

### During Call (Live Pipeline)

```
Audio chunks ──→ send_audio_to_gemini() ──→ _receive_loop() ──→ captions + audio out
                                                                   (critical path, no semaphore)

Video frames ──→ asyncio.create_task(analyze_frame_with_vision()) ──→ _vision_sem
             ├─→ asyncio.create_task(narrate_frame_with_gemini())  ──→ _gemini_sem
             └─→ send_video_to_gemini() (to Live session)
```

- Audio is the **critical path** — never blocked by vision/narration
- Vision API runs in `asyncio.to_thread()` so the synchronous gRPC call doesn't block the event loop
- Vision and narration run as fire-and-forget tasks, gated by semaphores
- All state mutations (screenshot/caption appends) go through `_state_lock`

### Post-Call (Memorabilia Pipeline)

```
POST /generate-all ──→ returns task_id immediately
                   └─→ asyncio.create_task(_run_parallel_generation())
                        ├── gen_storybook()  ──→ _gemini_sem ──→ Gemini interleaved output
                        ├── gen_video()      ──→ stylize (gather) ──→ Veo (poll)
                        └── gen_captions()   ──→ instant (read from archive)

Frontend polls GET /generation-status/{task_id} every 2s
  → shows each result as soon as its sub-task completes
```

- All three pipelines run concurrently via `asyncio.gather()`
- Storybook and video don't share intermediate work — truly independent
- Frontend shows results progressively (captions appear first, then storybook, then video)
- Semaphore prevents storybook + video from competing for Gemini quota

### Cloud Run Configuration

- **1 worker** (required: in-memory room state + WebSocket session affinity)
- **200 concurrent connections** per instance
- **120s keepalive timeout** for WebSocket longevity
- Deploy with `--session-affinity` flag

---

## Troubleshooting Quick Reference

| Symptom | Agent | Fix |
|---------|-------|-----|
| Gemini session dies immediately | Backend | Check `_session_context` lifecycle |
| No captions appearing | Backend | Verify `_receive_loop` is running, check `side` field |
| Audio plays but garbled | Frontend | Check sample rate match (16kHz vs 24kHz) |
| Vision never triggers | Backend | Check `_vision_allowed()`, ensure trigger phrase detected |
| Storybook fails with KeyError | Google API | Screenshot format mismatch — use `isinstance()` checks |
| Memory video timeout | Google API | Veo polling — increase max_wait or check image format |
| Camera PiP blank | Frontend | Check `getUserMedia` permissions, `camPip` display |
| High API costs | Orchestrator | Audit `log_api_call` paths, check polling intervals |
