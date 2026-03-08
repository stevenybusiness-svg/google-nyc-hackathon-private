"""
Storybook generator — transcript-driven illustrated storybook with audio narration.

Flow:
  1. Raw transcript → Gemini extracts story scenes
  2. Scenes → Gemini Interleaved Output → illustrated storybook pages
  3. Story text → Gemini TTS → audio narration per page
  4. Pages rendered as standalone HTML with embedded audio players
"""
import asyncio
import base64
import io
import json
import logging
import struct
import wave
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Modality

logger = logging.getLogger(__name__)

EXTRACT_SCENES_PROMPT = """You are a story editor. Given a transcript from a conversation where someone shares a story or experience, extract it into structured scenes.

TRANSCRIPT:
{transcript}

INSTRUCTIONS:
1. Identify the narrative (ignore small talk, greetings, interruptions)
2. Break the story into 3-6 key scenes/moments
3. For each scene, write:
   - A short title
   - The story text (rewritten to flow as an illustrated storybook page — warm, vivid, present tense)
   - A visual description for illustration (detailed enough for image generation)
   - A narration line (what a narrator would say aloud — 1-2 warm sentences)
   - Whether this scene is dramatic (the climax or most visually exciting moment)

Return ONLY valid JSON array:
[
  {{
    "title": "The Little River",
    "story_text": "Near a village nestled between green hills, a little river dances over smooth stones...",
    "visual_description": "A sparkling river winding through green hills with wildflowers on the banks, warm golden sunlight, watercolor illustration style",
    "narration": "Near a quiet village, a little river dances happily over smooth, shiny stones.",
    "dramatic": false
  }}
]

RULES:
- Keep the storyteller's voice and warmth
- Make visual descriptions rich and specific (colors, lighting, composition)
- Every visual description must end with: "watercolor illustration style"
- If no clear story is found, create scenes from the most narrative/descriptive parts
- Narration should be warm, gentle, and suitable for reading aloud
- 3-6 scenes maximum
- Mark exactly 1-2 scenes as "dramatic": true — the climax or most visually action-packed moments. The rest must be "dramatic": false
"""

STORYBOOK_PROMPT = """Create a richly illustrated storybook from these scenes. Generate multiple images per scene to show the story unfolding.

STORY TITLE: {title}
TOLD BY: {storyteller}

SCENES:
{scenes_text}

INSTRUCTIONS:
- For each scene, write the story text, then generate 2-3 illustrations showing different moments within that scene
- Illustrations should be warm, colorful, watercolor-style art
- Keep a consistent visual style across ALL illustrations (same characters, same color palette)
- The text should flow naturally from page to page
- Add a brief opening and a warm closing

FORMAT:
For each scene: write the text first, then generate 2-3 images showing the progression of that scene.
Example: Scene about a cloud hiding → Image 1: cloud looking scared, Image 2: cloud hiding behind mountain, Image 3: close-up of cloud's worried face.
Each text section should be 2-4 sentences. Do not describe the images in text — let the images speak for themselves.
"""


async def extract_story_scenes(client: genai.Client, transcript: str, storyteller: str = "Someone") -> list[dict]:
    """Extract structured story scenes from a raw transcript using Gemini."""
    try:
        prompt = EXTRACT_SCENES_PROMPT.format(transcript=transcript)
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=GenerateContentConfig(response_mime_type="application/json"),
        )
        text = response.text.strip()
        scenes = json.loads(text)
        if not isinstance(scenes, list) or len(scenes) == 0:
            logger.warning(f"Scene extraction returned invalid data: {text[:200]}")
            return []
        logger.info(f"Extracted {len(scenes)} scenes from transcript")
        return scenes
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse scenes JSON: {e}")
        try:
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
                return json.loads(text)
        except Exception:
            pass
        return []
    except Exception as e:
        logger.error(f"Scene extraction error: {type(e).__name__}: {e}")
        return []


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM audio data in a WAV container for browser playback."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


async def generate_narration_audio(
    client: genai.Client, text: str, voice_name: str = "Kore",
) -> dict | None:
    """Generate audio narration for a single page using Gemini TTS."""
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=f"Read this storybook page aloud with warmth and gentle expression:\n\n{text}",
            config=GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    ),
                ),
            ),
        )
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    audio_data = part.inline_data.data
                    if isinstance(audio_data, str):
                        audio_data = base64.b64decode(audio_data)
                    mime = part.inline_data.mime_type or ""
                    # Gemini TTS returns raw PCM — wrap in WAV for browser playback
                    if not mime.startswith("audio/wav") and not mime.startswith("audio/mp"):
                        audio_data = pcm_to_wav(audio_data, sample_rate=24000)
                        mime = "audio/wav"
                    return {"data": audio_data, "mime_type": mime}
        logger.warning("TTS returned no audio data")
        return None
    except Exception as e:
        logger.error(f"TTS error: {type(e).__name__}: {e}")
        return None


async def generate_all_narrations(
    client: genai.Client, pages: list, voice_name: str = "Kore",
) -> list[dict | None]:
    """Generate audio narration for all text pages concurrently."""
    tasks = []
    for page in pages:
        if page["type"] == "text" and page["content"].strip():
            tasks.append(generate_narration_audio(client, page["content"], voice_name))
    if not tasks:
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    narrations = []
    for r in results:
        if isinstance(r, dict):
            narrations.append(r)
        else:
            if isinstance(r, Exception):
                logger.warning(f"Narration task failed: {r}")
            narrations.append(None)
    return narrations


async def generate_storybook(client: genai.Client, storybook_input: dict) -> list:
    """Generate an illustrated storybook from transcript or pre-extracted scenes."""
    try:
        storyteller = storybook_input.get("storyteller", "Someone")
        title = storybook_input.get("title", "")
        screenshots = storybook_input.get("screenshots", [])[:3]

        scenes = storybook_input.get("scenes")
        if not scenes:
            transcript = storybook_input.get("transcript", "")
            if not transcript:
                snippets = storybook_input.get("voice_snippets", [])
                transcript = "\n".join(
                    s.get("text", "") for s in snippets if isinstance(s, dict)
                )
            if not transcript:
                logger.warning("No transcript or scenes provided")
                return []
            scenes = await extract_story_scenes(client, transcript, storyteller)
            if not scenes:
                logger.warning("Could not extract scenes from transcript")
                return []

        storybook_input["_extracted_scenes"] = scenes

        if not title and scenes:
            title = scenes[0].get("title", "A Story")

        scenes_text = ""
        for i, scene in enumerate(scenes, 1):
            scenes_text += f"\nScene {i}: {scene.get('title', f'Scene {i}')}\n"
            scenes_text += f"Story: {scene.get('story_text', '')}\n"
            scenes_text += f"Illustration: {scene.get('visual_description', '')}\n"

        prompt = STORYBOOK_PROMPT.format(title=title, storyteller=storyteller, scenes_text=scenes_text)

        contents = []
        for screenshot in screenshots:
            img_data = None
            if isinstance(screenshot, dict):
                img_data = screenshot.get("data", "")
            elif isinstance(screenshot, str):
                img_data = screenshot
            if img_data:
                if isinstance(img_data, bytes):
                    img_data = base64.b64encode(img_data).decode()
                contents.append({"inline_data": {"mime_type": "image/jpeg", "data": img_data}})
        contents.append({"text": prompt})

        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=GenerateContentConfig(response_modalities=[Modality.TEXT, Modality.IMAGE]),
        )

        pages = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    pages.append({"type": "text", "content": part.text})
                elif hasattr(part, "inline_data") and part.inline_data:
                    img = part.inline_data.data
                    if isinstance(img, str):
                        img = base64.b64decode(img)
                    pages.append({"type": "image", "content": img, "mime_type": part.inline_data.mime_type or "image/png"})

        if not pages:
            logger.warning("Storybook generation returned no content")
        else:
            logger.info(f"Generated storybook: {len(pages)} pages "
                        f"({sum(1 for p in pages if p['type'] == 'text')} text, "
                        f"{sum(1 for p in pages if p['type'] == 'image')} images)")
        return pages
    except Exception as e:
        logger.error(f"Error generating storybook: {type(e).__name__}: {e}")
        return []


def get_storybook_images(pages: list) -> list[dict]:
    """Extract images from storybook pages for use in video pipeline."""
    images = []
    for page in pages:
        if page["type"] == "image":
            img_data = page["content"]
            if isinstance(img_data, str):
                img_data = base64.b64decode(img_data)
            images.append({"data": base64.b64encode(img_data).decode(), "mime_type": page.get("mime_type", "image/png")})
    return images


def render_storybook_html(pages: list, title: str = "Our Story", audio_narrations: list | None = None) -> str:
    """Render storybook pages to HTML with optional embedded audio players."""
    html_parts = [f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@300;400&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Georgia','Fraunces',serif;max-width:800px;margin:0 auto;padding:40px 20px;background:#faf8f5;color:#333}}
h1{{text-align:center;font-weight:300;color:#5a4a3a;margin-bottom:2rem;font-size:2rem}}
.page-text{{font-size:1.15rem;line-height:1.9;margin:20px 0;color:#4a3a2a}}
.page-image{{width:100%;border-radius:14px;box-shadow:0 6px 24px rgba(0,0,0,0.08);margin:16px 0}}
.audio-player{{display:flex;align-items:center;gap:10px;margin:12px 0;padding:10px 16px;background:#f0ebe4;border-radius:24px;width:fit-content}}
.play-btn{{width:36px;height:36px;border-radius:50%;border:none;background:#5a4a3a;color:white;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:14px;transition:background 0.2s}}
.play-btn:hover{{background:#4a3a2a}}.play-btn.playing{{background:#c0392b}}
.audio-label{{font-size:0.82rem;color:#8a7a6a;font-family:'DM Sans',sans-serif}}
.divider{{height:1px;background:linear-gradient(90deg,transparent,#d4c4b4,transparent);margin:2rem 0}}
</style></head><body><h1>{title}</h1>
"""]

    text_audio_idx = 0
    for i, page in enumerate(pages):
        if page["type"] == "text":
            html_parts.append(f'<div class="page-text">{page["content"]}</div>')
            if audio_narrations and text_audio_idx < len(audio_narrations):
                narration = audio_narrations[text_audio_idx]
                if narration and isinstance(narration, dict):
                    audio_b64 = narration["data"]
                    if isinstance(audio_b64, bytes):
                        audio_b64 = base64.b64encode(audio_b64).decode()
                    mime = narration.get("mime_type", "audio/mpeg")
                    aid = f"audio_{i}"
                    html_parts.append(f'<div class="audio-player"><button class="play-btn" onclick="toggleAudio(\'{aid}\',this)" id="btn_{aid}">&#9654;</button><audio id="{aid}" src="data:{mime};base64,{audio_b64}" onended="document.getElementById(\'btn_{aid}\').innerHTML=\'&#9654;\';document.getElementById(\'btn_{aid}\').classList.remove(\'playing\')"></audio><span class="audio-label">Listen</span></div>')
                text_audio_idx += 1
        elif page["type"] == "image":
            raw = page["content"]
            if isinstance(raw, bytes):
                img_data = base64.b64encode(raw).decode()
            elif isinstance(raw, str):
                img_data = raw
            else:
                continue
            mime = page.get("mime_type", "image/png")
            html_parts.append(f'<img class="page-image" src="data:{mime};base64,{img_data}">')
            html_parts.append('<div class="divider"></div>')

    html_parts.append("""<script>
function toggleAudio(id,btn){var a=document.getElementById(id);if(a.paused){document.querySelectorAll('audio').forEach(x=>{x.pause();x.currentTime=0});document.querySelectorAll('.play-btn').forEach(b=>{b.innerHTML='&#9654;';b.classList.remove('playing')});a.play();btn.innerHTML='&#9724;';btn.classList.add('playing')}else{a.pause();a.currentTime=0;btn.innerHTML='&#9654;';btn.classList.remove('playing')}}
</script></body></html>""")
    return "\n".join(html_parts)
