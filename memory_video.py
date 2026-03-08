"""
Memory Video pipeline — stylize screenshots and generate a video keepsake.

Flow:
  1. Screenshots → Gemini image generation (stylized, warm, memory-like)
  2. Stylized images → Veo 3.1 Fast → short video
"""
import asyncio
import base64
import logging
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

VIDEO_PROMPT_TEMPLATE = """Create a slow, contemplative video montage from these stylized memory images.

CONTEXT: These images are from a video call between {participants} — two loved ones separated by distance, connected through translation technology.

STYLE:
- Slow Ken Burns effect (gentle pan and zoom on each image)
- Warm color grading throughout
- Soft crossfade transitions between images
- Contemplative, emotional pace
- The feeling of looking through a cherished photo album

MOOD: Love, warmth, nostalgia, connection despite distance

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
    participants: str = "two loved ones",
) -> dict | None:
    """Generate a short memory video from stylized images using Veo."""
    if not stylized_images:
        logger.warning("No stylized images to generate video from")
        return None

    try:
        prompt = VIDEO_PROMPT_TEMPLATE.format(participants=participants)

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


async def run_memory_video_pipeline(
    client: genai.Client,
    screenshots: list[dict],
    participants: str = "two loved ones",
) -> dict:
    """Full pipeline: stylize → generate video. Returns result dict."""
    result = {
        "stylized_images": [],
        "video": None,
    }

    logger.info(f"Stylizing {len(screenshots)} screenshots...")
    stylized = await stylize_all_images(client, screenshots)
    result["stylized_images"] = stylized
    logger.info(f"Stylized {len(stylized)} images")

    if stylized:
        logger.info("Generating memory video...")
        video = await generate_memory_video(client, stylized, participants)
        result["video"] = video
        if video:
            logger.info("Memory video generated successfully")

    return result
