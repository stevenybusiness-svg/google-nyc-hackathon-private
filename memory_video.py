"""
Memory Video pipeline — stylize screenshots and generate a video keepsake.

Flow:
  1. Screenshots → Gemini image generation (stylized, warm, memory-like)
  2. Stylized images → Veo 3.1 Fast → short video
"""
import asyncio
import base64
import logging
import os
import subprocess
import tempfile
import time

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

STYLIZE_PROMPT = """Transform this photograph into a warm, dreamlike, artistic illustration.

STYLE:
- Soft warm lighting, golden hour feel
- Gentle watercolor or oil painting aesthetic
- Dreamy, slightly blurred edges
- Emphasize warmth, love, and human connection
- Muted but warm color palette (amber, soft pink, cream)

CONSTRAINTS:
- No text or watermarks
- No harsh lighting or shadows
- Keep the emotional essence of the original scene
- The result should feel like a treasured memory, not a photograph
"""

VIDEO_PROMPT_TEMPLATE = """Animate this storybook illustration into a gentle, dreamy scene.

CONTEXT: {context}

STYLE:
- Slow Ken Burns effect (gentle pan and zoom)
- Warm color grading, golden hour lighting
- Soft particle effects (floating dust motes, gentle light rays)
- Watercolor art style maintained throughout
- The feeling of a storybook page coming to life

MOOD: Warm, magical, nostalgic

Duration: ~8 seconds
"""


async def stylize_image(client: genai.Client, image_b64: str) -> dict | None:
    """Stylize a single screenshot using Gemini image generation."""
    try:
        if isinstance(image_b64, bytes):
            image_b64 = base64.b64encode(image_b64).decode()

        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_b64,
                    }
                },
                {"text": STYLIZE_PROMPT},
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    img_data = part.inline_data.data
                    if isinstance(img_data, str):
                        img_data = base64.b64decode(img_data)
                    return {
                        "data": img_data,
                        "mime_type": part.inline_data.mime_type or "image/png",
                    }
        logger.warning("Stylization returned no image data")
        return None
    except Exception as e:
        logger.error(f"Image stylization error: {e}")
        return None


async def stylize_all_images(
    client: genai.Client, screenshots: list, max_images: int = 5
) -> list[dict]:
    """Stylize multiple screenshots concurrently."""
    tasks = []
    for ss in screenshots[:max_images]:
        if isinstance(ss, dict):
            img_data = ss.get("data", "")
        elif isinstance(ss, str):
            img_data = ss
        else:
            continue
        if img_data:
            tasks.append(stylize_image(client, img_data))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    stylized = []
    for r in results:
        if isinstance(r, dict) and r is not None:
            stylized.append(r)
    return stylized


async def generate_memory_video(
    client: genai.Client,
    stylized_images: list[dict],
    context: str = "A storybook illustration from a tale told by a grandparent to their grandchild",
) -> dict | None:
    """Generate a short memory video from storybook/stylized images using Veo."""
    if not stylized_images:
        logger.warning("No stylized images to generate video from")
        return None

    try:
        prompt = VIDEO_PROMPT_TEMPLATE.format(context=context)

        image_part = stylized_images[0]
        img_bytes = image_part["data"]
        if isinstance(img_bytes, str):
            try:
                img_bytes = base64.b64decode(img_bytes)
            except Exception:
                logger.error("Failed to decode stylized image data from base64")
                return None

        if not isinstance(img_bytes, bytes) or len(img_bytes) < 100:
            logger.error(f"Invalid image bytes: type={type(img_bytes)}, len={len(img_bytes) if hasattr(img_bytes, '__len__') else '?'}")
            return None

        mime = image_part.get("mime_type", "image/png")
        logger.info(f"Generating video from {len(img_bytes)} bytes ({mime})")

        operation = await client.aio.models.generate_videos(
            model="veo-2.0-generate-001",
            prompt=prompt,
            image=types.Image(
                image_bytes=img_bytes,
                mime_type=mime,
            ),
            config=types.GenerateVideosConfig(
                number_of_videos=1,
                duration_seconds=8,
                person_generation="allow_all",
            ),
        )

        max_wait = 300
        start = time.time()
        while not operation.done and (time.time() - start) < max_wait:
            elapsed = int(time.time() - start)
            logger.info(f"Video generation polling... {elapsed}s elapsed")
            await asyncio.sleep(5)
            try:
                operation = await client.aio.operations.get(operation)
            except Exception as poll_err:
                logger.warning(f"Polling error (will retry): {poll_err}")
                await asyncio.sleep(3)

        if not operation.done:
            logger.warning(f"Video generation timed out after {max_wait}s")
            return None

        if operation.result and operation.result.generated_videos:
            video = operation.result.generated_videos[0]
            # Vertex AI returns video_bytes directly; API key mode uses files.download
            if hasattr(video.video, 'video_bytes') and video.video.video_bytes:
                video_data = video.video.video_bytes
            else:
                video_data = await client.aio.files.download(file=video.video)
            logger.info(f"Video downloaded: {len(video_data) if video_data else 0} bytes")
            return {
                "data": video_data,
                "mime_type": "video/mp4",
            }

        logger.warning(f"Video generation completed but no videos returned. Result: {operation.result}")
        return None
    except Exception as e:
        logger.error(f"Video generation error: {type(e).__name__}: {e}")
        return None


def combine_video_audio(video_bytes: bytes, audio_bytes: bytes, audio_mime: str = "audio/wav") -> bytes | None:
    """Combine a silent video with narration audio using ffmpeg.

    Returns the combined MP4 bytes, or None if ffmpeg fails.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            audio_ext = "wav" if "wav" in audio_mime else "mp3"
            audio_path = os.path.join(tmpdir, f"audio.{audio_ext}")
            output_path = os.path.join(tmpdir, "output.mp4")

            with open(video_path, "wb") as f:
                f.write(video_bytes)
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            # Combine: use video stream from video, audio stream from narration
            # -shortest: end when the shorter stream ends (video is ~8s)
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest",
                "-movflags", "+faststart",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr.decode()[:500]}")
                return None

            with open(output_path, "rb") as f:
                return f.read()
    except Exception as e:
        logger.error(f"Video+audio combine error: {type(e).__name__}: {e}")
        return None


def concatenate_audio(audio_list: list[dict]) -> bytes | None:
    """Concatenate multiple WAV audio narrations into one WAV file using ffmpeg."""
    if not audio_list:
        return None
    if len(audio_list) == 1:
        return audio_list[0]["data"]
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write each audio file
            paths = []
            for i, narration in enumerate(audio_list):
                ext = "wav" if "wav" in narration.get("mime_type", "") else "mp3"
                path = os.path.join(tmpdir, f"part_{i}.{ext}")
                with open(path, "wb") as f:
                    f.write(narration["data"])
                paths.append(path)

            # Create ffmpeg concat file
            concat_path = os.path.join(tmpdir, "concat.txt")
            with open(concat_path, "w") as f:
                for p in paths:
                    f.write(f"file '{p}'\n")

            output_path = os.path.join(tmpdir, "combined.wav")
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_path,
                "-c:a", "pcm_s16le",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"Audio concat failed: {result.stderr.decode()[:500]}")
                return None

            with open(output_path, "rb") as f:
                return f.read()
    except Exception as e:
        logger.error(f"Audio concatenation error: {type(e).__name__}: {e}")
        return None


# ─── Scene-by-scene narrated story video (Ken Burns + ffmpeg) ───

# Ken Burns effects: rotate through these for visual variety
KEN_BURNS_EFFECTS = [
    # Slow zoom in (center)
    "zoompan=z='min(zoom+0.0008,1.25)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720:fps=24",
    # Pan left to right
    "zoompan=z=1.15:x='(iw-iw/zoom)/2 + (iw/zoom)*0.15*on/{frames}':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720:fps=24",
    # Pan right to left
    "zoompan=z=1.15:x='(iw-iw/zoom)/2 - (iw/zoom)*0.15*on/{frames}':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720:fps=24",
    # Slow zoom out
    "zoompan=z='if(eq(on,1),1.25,max(zoom-0.0008,1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720:fps=24",
]


def get_audio_duration(audio_bytes: bytes) -> float:
    """Get duration of WAV audio in seconds from header.

    Assumes 24kHz, 16-bit mono WAV (our pcm_to_wav output format).
    Falls back to ffprobe if header parsing fails.
    """
    # WAV header: 44 bytes, rest is PCM data
    # duration = (data_bytes) / (sample_rate * channels * bytes_per_sample)
    if len(audio_bytes) > 44 and audio_bytes[:4] == b'RIFF':
        data_size = len(audio_bytes) - 44
        return data_size / (24000 * 1 * 2)  # 24kHz, mono, 16-bit
    # Fallback: use ffprobe
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            f.flush()
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
                   "-show_format", f.name]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            os.unlink(f.name)
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                return float(info["format"]["duration"])
    except Exception:
        pass
    return 5.0  # safe default


def _create_single_image_clip(
    image_path: str, duration: float, effect_index: int, output_path: str,
) -> bool:
    """Create a silent Ken Burns clip from a single image."""
    fps = 24
    frames = int(duration * fps)
    effect_template = KEN_BURNS_EFFECTS[effect_index % len(KEN_BURNS_EFFECTS)]
    effect = effect_template.format(frames=frames)

    filter_str = (
        f"[0:v]scale=1920:1080:force_original_aspect_ratio=increase,"
        f"crop=1920:1080,{effect},format=yuv420p[v]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-filter_complex", filter_str,
        "-map", "[v]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-t", str(duration),
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False


def create_ken_burns_clip(
    images: list[bytes], audio_bytes: bytes, duration: float,
    scene_index: int, tmpdir: str,
) -> str | None:
    """Create a scene video clip with Ken Burns effects across multiple images + audio.

    If multiple images: each image gets equal time with crossfade transitions.
    Returns the output file path, or None on failure.
    """
    audio_path = os.path.join(tmpdir, f"audio_{scene_index}.wav")
    output_path = os.path.join(tmpdir, f"clip_{scene_index}.mp4")

    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    num_images = len(images)
    crossfade_dur = 0.8  # seconds of crossfade between images

    if num_images == 1:
        # Simple case: one image for the whole scene
        image_path = os.path.join(tmpdir, f"scene_{scene_index}_0.png")
        with open(image_path, "wb") as f:
            f.write(images[0])

        fps = 24
        frames = int(duration * fps)
        effect_template = KEN_BURNS_EFFECTS[scene_index % len(KEN_BURNS_EFFECTS)]
        effect = effect_template.format(frames=frames)
        filter_str = (
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=increase,"
            f"crop=1920:1080,{effect},format=yuv420p[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", image_path,
            "-i", audio_path,
            "-filter_complex", filter_str,
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0:
                logger.error(f"Ken Burns clip {scene_index} failed: {result.stderr.decode()[:500]}")
                return None
            return output_path
        except Exception as e:
            logger.error(f"Ken Burns clip error: {type(e).__name__}: {e}")
            return None

    # Multiple images: create sub-clips, then crossfade them together + add audio
    total_crossfade = crossfade_dur * (num_images - 1)
    per_image_dur = (duration + total_crossfade) / num_images

    sub_clips = []
    for i, img_bytes in enumerate(images):
        img_path = os.path.join(tmpdir, f"scene_{scene_index}_{i}.png")
        sub_path = os.path.join(tmpdir, f"scene_{scene_index}_sub_{i}.mp4")
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        effect_idx = (scene_index * num_images + i) % len(KEN_BURNS_EFFECTS)
        if _create_single_image_clip(img_path, per_image_dur, effect_idx, sub_path):
            sub_clips.append(sub_path)
        else:
            logger.warning(f"Sub-clip {scene_index}_{i} failed, skipping")

    if not sub_clips:
        return None

    if len(sub_clips) == 1:
        # Only one sub-clip succeeded, just add audio
        merged_path = os.path.join(tmpdir, f"scene_{scene_index}_merged.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", sub_clips[0], "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-shortest", "-movflags", "+faststart", merged_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0:
                os.rename(merged_path, output_path)
                return output_path
        except Exception:
            pass
        return None

    # Build xfade filter chain for crossfading between sub-clips
    inputs = []
    for sc in sub_clips:
        inputs.extend(["-i", sc])

    filter_parts = []
    n = len(sub_clips)

    # First xfade
    offset = per_image_dur - crossfade_dur
    filter_parts.append(
        f"[0:v][1:v]xfade=transition=fade:duration={crossfade_dur}:offset={offset:.3f}[v01]"
    )
    cumulative_dur = 2 * per_image_dur - crossfade_dur

    for i in range(2, n):
        prev_label = f"v{i-2}{i-1}" if i == 2 else f"v_acc{i-1}"
        next_label = f"v_acc{i}" if i < n - 1 else "vfinal"
        offset = cumulative_dur - crossfade_dur
        filter_parts.append(
            f"[{prev_label}][{i}:v]xfade=transition=fade:duration={crossfade_dur}:offset={offset:.3f}[{next_label}]"
        )
        cumulative_dur += per_image_dur - crossfade_dur

    final_video_label = "vfinal" if n > 2 else "v01"

    filter_str = ";".join(filter_parts)
    xfade_path = os.path.join(tmpdir, f"scene_{scene_index}_xfade.mp4")

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_str,
        "-map", f"[{final_video_label}]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        xfade_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            logger.error(f"xfade scene {scene_index} failed: {result.stderr.decode()[:500]}")
            # Fallback: simple concat without crossfade
            concat_path = os.path.join(tmpdir, f"scene_{scene_index}_concat.txt")
            with open(concat_path, "w") as f:
                for sc in sub_clips:
                    f.write(f"file '{sc}'\n")
            cmd_fallback = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_path, "-c", "copy", xfade_path,
            ]
            result = subprocess.run(cmd_fallback, capture_output=True, timeout=60)
            if result.returncode != 0:
                return None
    except Exception as e:
        logger.error(f"xfade error: {type(e).__name__}: {e}")
        return None

    # Add audio to the crossfaded video
    cmd = [
        "ffmpeg", "-y",
        "-i", xfade_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"Audio mux scene {scene_index} failed: {result.stderr.decode()[:500]}")
            return None
        return output_path
    except Exception as e:
        logger.error(f"Audio mux error: {type(e).__name__}: {e}")
        return None


def pair_scenes_with_media(
    pages: list, narrations: list[dict | None], scenes: list[dict],
) -> list[dict]:
    """Bridge storybook output into scene-by-scene video input.

    Maps interleaved pages (text, image..., text, image...) + narrations (aligned
    to text pages) into paired dicts with multiple images per scene.
    """
    paired = []
    text_idx = 0
    current_narration = None
    current_images = []

    def flush_scene():
        nonlocal current_narration, current_images
        if current_narration and current_images:
            scene_idx = len(paired)
            scene_meta = scenes[scene_idx] if scene_idx < len(scenes) else {}
            paired.append({
                "images": list(current_images),
                "audio": current_narration["data"],
                "audio_mime": current_narration.get("mime_type", "audio/wav"),
                "title": scene_meta.get("title", f"Scene {scene_idx + 1}"),
                "dramatic": scene_meta.get("dramatic", False),
                "visual_description": scene_meta.get("visual_description", ""),
            })
        current_narration = None
        current_images = []

    for page in pages:
        if page["type"] == "text":
            # New text section — flush previous scene if it had images
            flush_scene()
            if narrations and text_idx < len(narrations):
                current_narration = narrations[text_idx]
            text_idx += 1
        elif page["type"] == "image":
            img_data = page["content"]
            if isinstance(img_data, str):
                img_data = base64.b64decode(img_data)
            current_images.append(img_data)

    # Flush last scene
    flush_scene()
    return paired


def create_story_video(scenes_with_media: list[dict]) -> dict | None:
    """Create a full narrated story video from scene images + audio narrations.

    Each scene gets a Ken Burns animated clip (with crossfades between multiple
    images) matching its narration duration, then all clips are concatenated.

    Args:
        scenes_with_media: list of {"images": [bytes, ...], "audio": bytes, "audio_mime": str, "title": str}

    Returns:
        {"data": bytes, "mime_type": "video/mp4", "duration_seconds": float} or None
    """
    if not scenes_with_media:
        logger.warning("No scenes to create story video from")
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_paths = []
            total_duration = 0.0

            for i, scene in enumerate(scenes_with_media):
                duration = get_audio_duration(scene["audio"])
                images = scene.get("images", [])
                logger.info(f"Scene {i}: '{scene.get('title', '?')}' — {duration:.1f}s, {len(images)} images")

                clip_path = create_ken_burns_clip(
                    images, scene["audio"], duration, i, tmpdir,
                )
                if clip_path:
                    clip_paths.append(clip_path)
                    total_duration += duration
                else:
                    logger.warning(f"Skipping scene {i} — clip creation failed")

            if not clip_paths:
                logger.error("No clips created, cannot produce video")
                return None

            # Concatenate all clips
            if len(clip_paths) == 1:
                with open(clip_paths[0], "rb") as f:
                    return {
                        "data": f.read(),
                        "mime_type": "video/mp4",
                        "duration_seconds": total_duration,
                    }

            concat_path = os.path.join(tmpdir, "concat.txt")
            with open(concat_path, "w") as f:
                for p in clip_paths:
                    f.write(f"file '{p}'\n")

            output_path = os.path.join(tmpdir, "story_video.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_path,
                "-c", "copy",
                "-movflags", "+faststart",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0:
                logger.error(f"Video concat failed: {result.stderr.decode()[:500]}")
                return None

            with open(output_path, "rb") as f:
                video_data = f.read()

            logger.info(f"Story video created: {len(clip_paths)} scenes, "
                        f"{total_duration:.1f}s, {len(video_data)} bytes")
            return {
                "data": video_data,
                "mime_type": "video/mp4",
                "duration_seconds": total_duration,
            }
    except Exception as e:
        logger.error(f"Story video creation error: {type(e).__name__}: {e}")
        return None


async def generate_scene_veo_video(
    client: genai.Client, image_bytes: bytes, scene_description: str,
) -> dict | None:
    """Generate an 8s Veo video from a single storybook image + scene context."""
    try:
        prompt = VIDEO_PROMPT_TEMPLATE.format(context=scene_description)
        mime = "image/png"

        if isinstance(image_bytes, str):
            image_bytes = base64.b64decode(image_bytes)

        if not isinstance(image_bytes, bytes) or len(image_bytes) < 100:
            logger.error("Invalid image for Veo scene video")
            return None

        logger.info(f"Veo: generating 8s video for scene ({len(image_bytes)} bytes)")
        operation = await client.aio.models.generate_videos(
            model="veo-2.0-generate-001",
            prompt=prompt,
            image=types.Image(image_bytes=image_bytes, mime_type=mime),
            config=types.GenerateVideosConfig(
                number_of_videos=1,
                duration_seconds=8,
                person_generation="allow_all",
            ),
        )

        max_wait = 300
        start = time.time()
        while not operation.done and (time.time() - start) < max_wait:
            elapsed = int(time.time() - start)
            logger.info(f"Veo scene polling... {elapsed}s")
            await asyncio.sleep(5)
            try:
                operation = await client.aio.operations.get(operation)
            except Exception as poll_err:
                logger.warning(f"Veo poll error: {poll_err}")
                await asyncio.sleep(3)

        if not operation.done:
            logger.warning(f"Veo scene timed out after {max_wait}s")
            return None

        if operation.result and operation.result.generated_videos:
            video = operation.result.generated_videos[0]
            if hasattr(video.video, 'video_bytes') and video.video.video_bytes:
                video_data = video.video.video_bytes
            else:
                video_data = await client.aio.files.download(file=video.video)
            logger.info(f"Veo scene video: {len(video_data)} bytes")
            return {"data": video_data, "mime_type": "video/mp4"}

        logger.warning("Veo scene returned no video")
        return None
    except Exception as e:
        logger.error(f"Veo scene error: {type(e).__name__}: {e}")
        return None


def _make_veo_scene_clip(
    veo_video_bytes: bytes, audio_bytes: bytes, narration_duration: float,
    scene_index: int, images: list[bytes], tmpdir: str,
) -> str | None:
    """Create a scene clip from Veo video + narration audio.

    If narration > 8s, use Veo for first 8s then Ken Burns for the rest.
    If narration <= 8s, trim Veo to narration duration.
    """
    veo_path = os.path.join(tmpdir, f"veo_{scene_index}.mp4")
    audio_path = os.path.join(tmpdir, f"veo_audio_{scene_index}.wav")
    output_path = os.path.join(tmpdir, f"clip_{scene_index}.mp4")

    with open(veo_path, "wb") as f:
        f.write(veo_video_bytes)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    veo_duration = 8.0  # Veo always generates 8s

    if narration_duration <= veo_duration + 0.5:
        # Narration fits within Veo video — just trim and add audio
        cmd = [
            "ffmpeg", "-y",
            "-i", veo_path, "-i", audio_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-t", str(narration_duration),
            "-movflags", "+faststart",
            output_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0:
                return output_path
            logger.error(f"Veo trim failed: {result.stderr.decode()[:300]}")
        except Exception as e:
            logger.error(f"Veo trim error: {e}")
        return None

    # Narration longer than Veo — Veo for 8s, then Ken Burns for remainder
    remaining = narration_duration - veo_duration + 0.8  # overlap for crossfade

    # Scale Veo to 1280x720 to match Ken Burns output
    veo_scaled_path = os.path.join(tmpdir, f"veo_{scene_index}_scaled.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", veo_path,
        "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,format=yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-an", veo_scaled_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"Veo scale failed: {result.stderr.decode()[:300]}")
            return None
    except Exception:
        return None

    # Create Ken Burns clip for the remaining time (use last image)
    kb_image_path = os.path.join(tmpdir, f"veo_kb_{scene_index}.png")
    with open(kb_image_path, "wb") as f:
        f.write(images[-1] if images else images[0])

    kb_path = os.path.join(tmpdir, f"veo_kb_{scene_index}.mp4")
    if not _create_single_image_clip(kb_image_path, remaining, scene_index + 2, kb_path):
        # Fallback: just use trimmed Veo
        cmd = [
            "ffmpeg", "-y",
            "-i", veo_path, "-i", audio_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=60)
        return output_path if os.path.exists(output_path) else None

    # Crossfade Veo → Ken Burns
    xfade_offset = veo_duration - 0.8
    xfade_path = os.path.join(tmpdir, f"veo_xfade_{scene_index}.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", veo_scaled_path, "-i", kb_path,
        "-filter_complex",
        f"[0:v][1:v]xfade=transition=fade:duration=0.8:offset={xfade_offset:.3f}[v]",
        "-map", "[v]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        xfade_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            logger.error(f"Veo xfade failed: {result.stderr.decode()[:300]}")
            # Fallback: concat without crossfade
            concat_path = os.path.join(tmpdir, f"veo_concat_{scene_index}.txt")
            with open(concat_path, "w") as f:
                f.write(f"file '{veo_scaled_path}'\nfile '{kb_path}'\n")
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                   "-i", concat_path, "-c", "copy", xfade_path]
            subprocess.run(cmd, capture_output=True, timeout=60)
    except Exception:
        return None

    if not os.path.exists(xfade_path):
        return None

    # Add audio
    cmd = [
        "ffmpeg", "-y",
        "-i", xfade_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0:
            return output_path
    except Exception:
        pass
    return None


async def create_story_video_hybrid(
    client: genai.Client,
    scenes_with_media: list[dict],
) -> dict | None:
    """Create a narrated story video with Veo animations for dramatic scenes.

    Dramatic scenes get real Veo-generated video. Other scenes use Ken Burns.
    Falls back to Ken Burns if Veo fails for any scene.
    """
    if not scenes_with_media:
        return None

    # Identify dramatic scenes (max 2 for budget)
    dramatic_indices = [
        i for i, s in enumerate(scenes_with_media) if s.get("dramatic")
    ][:2]

    if not dramatic_indices:
        logger.info("No dramatic scenes flagged, using Ken Burns for all")
        return create_story_video(scenes_with_media)

    logger.info(f"Dramatic scenes: {dramatic_indices} — generating Veo videos")

    # Kick off Veo generation for dramatic scenes concurrently
    veo_tasks = {}
    for idx in dramatic_indices:
        scene = scenes_with_media[idx]
        # Use first image from the scene
        img = scene["images"][0] if scene["images"] else None
        if img:
            desc = scene.get("visual_description", scene.get("title", "A storybook scene"))
            veo_tasks[idx] = asyncio.create_task(
                generate_scene_veo_video(client, img, desc)
            )

    # While Veo runs, create Ken Burns clips for non-dramatic scenes
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_paths = [None] * len(scenes_with_media)
            total_duration = 0.0
            durations = []

            for i, scene in enumerate(scenes_with_media):
                dur = get_audio_duration(scene["audio"])
                durations.append(dur)

                if i not in dramatic_indices:
                    logger.info(f"Scene {i}: '{scene.get('title', '?')}' — {dur:.1f}s (Ken Burns, {len(scene['images'])} images)")
                    clip_path = create_ken_burns_clip(
                        scene["images"], scene["audio"], dur, i, tmpdir,
                    )
                    clip_paths[i] = clip_path
                    if clip_path:
                        total_duration += dur

            # Await Veo results
            for idx, task in veo_tasks.items():
                scene = scenes_with_media[idx]
                dur = durations[idx]
                try:
                    veo_result = await task
                except Exception as e:
                    logger.warning(f"Veo task {idx} raised: {e}")
                    veo_result = None

                if veo_result and veo_result.get("data"):
                    logger.info(f"Scene {idx}: '{scene.get('title', '?')}' — {dur:.1f}s (VEO + narration)")
                    clip_path = _make_veo_scene_clip(
                        veo_result["data"], scene["audio"], dur,
                        idx, scene["images"], tmpdir,
                    )
                    if clip_path:
                        clip_paths[idx] = clip_path
                        total_duration += dur
                        continue

                # Veo failed — fallback to Ken Burns
                logger.warning(f"Scene {idx}: Veo failed, falling back to Ken Burns")
                clip_path = create_ken_burns_clip(
                    scene["images"], scene["audio"], dur, idx, tmpdir,
                )
                clip_paths[idx] = clip_path
                if clip_path:
                    total_duration += dur

            # Filter out None entries
            valid_clips = [p for p in clip_paths if p]
            if not valid_clips:
                logger.error("No clips created")
                return None

            if len(valid_clips) == 1:
                with open(valid_clips[0], "rb") as f:
                    return {
                        "data": f.read(),
                        "mime_type": "video/mp4",
                        "duration_seconds": total_duration,
                    }

            # Concatenate all clips
            concat_path = os.path.join(tmpdir, "concat.txt")
            with open(concat_path, "w") as f:
                for p in valid_clips:
                    f.write(f"file '{p}'\n")

            output_path = os.path.join(tmpdir, "story_video.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_path,
                "-c", "copy",
                "-movflags", "+faststart",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0:
                logger.error(f"Final concat failed: {result.stderr.decode()[:500]}")
                return None

            with open(output_path, "rb") as f:
                video_data = f.read()

            veo_count = sum(1 for idx in dramatic_indices if clip_paths[idx] and idx in veo_tasks)
            logger.info(f"Hybrid story video: {len(valid_clips)} scenes "
                        f"({veo_count} Veo, {len(valid_clips) - veo_count} Ken Burns), "
                        f"{total_duration:.1f}s, {len(video_data)} bytes")
            return {
                "data": video_data,
                "mime_type": "video/mp4",
                "duration_seconds": total_duration,
            }
    except Exception as e:
        logger.error(f"Hybrid story video error: {type(e).__name__}: {e}")
        return None


async def run_memory_video_pipeline(
    client: genai.Client,
    images: list[dict],
    context: str = "A storybook illustration from a tale told by a grandparent",
    skip_stylize: bool = False,
    narrations: list[dict | None] | None = None,
) -> dict:
    """Full pipeline: optionally stylize → generate video → optionally add audio.

    Args:
        images: list of image dicts with "data" (base64 str or bytes) and "mime_type"
        context: narrative context for the video prompt
        skip_stylize: if True, images are already stylized (e.g. from storybook generation)
        narrations: optional list of audio narration dicts to combine with video
    """
    result = {
        "stylized_images": [],
        "video": None,
    }

    if skip_stylize:
        result["stylized_images"] = images
        stylized = images
        logger.info(f"Using {len(images)} pre-styled images (skipping stylization)")
    else:
        logger.info(f"Stylizing {len(images)} images...")
        stylized = await stylize_all_images(client, images)
        result["stylized_images"] = stylized
        logger.info(f"Stylized {len(stylized)} images")

    if stylized:
        logger.info("Generating memory video...")
        video = await generate_memory_video(client, stylized, context)
        result["video"] = video
        if video:
            logger.info("Memory video generated successfully")

            # Combine video with narration audio if available
            if narrations:
                valid_narrations = [n for n in narrations if n and isinstance(n, dict)]
                if valid_narrations:
                    logger.info(f"Combining video with {len(valid_narrations)} narration(s)...")
                    combined_audio = concatenate_audio(valid_narrations)
                    if combined_audio:
                        combined = combine_video_audio(
                            video["data"], combined_audio,
                            audio_mime=valid_narrations[0].get("mime_type", "audio/wav"),
                        )
                        if combined:
                            result["video_with_audio"] = {"data": combined, "mime_type": "video/mp4"}
                            logger.info(f"Video+audio combined: {len(combined)} bytes")
                        else:
                            logger.warning("Failed to combine video+audio, silent video still available")

    return result
