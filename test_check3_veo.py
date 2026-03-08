#!/usr/bin/env python3
"""
CHECK 3: Veo Video Generation
- Sends a stylized image + narrative prompt to veo-2.0-generate-001
- Expects an 8-second MP4 video back
- Saves: test_output/memory_video.mp4
- NOTE: This is the slowest check (~60-180s) and costs ~$2.40
"""
import asyncio
import base64
import io
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from memory_video import generate_memory_video


def make_stylized_test_image():
    """Create a warm, painterly-looking test image to feed Veo."""
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.new("RGB", (512, 512), (220, 180, 140))
    draw = ImageDraw.Draw(img)

    # Warm sunset sky gradient
    for y in range(200):
        r = int(220 + (255 - 220) * (1 - y / 200))
        g = int(140 + (180 - 140) * (1 - y / 200))
        b = int(80 + (120 - 80) * (1 - y / 200))
        draw.line([(0, y), (512, y)], fill=(min(r, 255), min(g, 255), min(b, 255)))

    # Ground
    draw.rectangle([0, 200, 512, 512], fill=(100, 130, 80))

    # Simple tree
    draw.rectangle([230, 120, 260, 300], fill=(120, 80, 50))
    draw.ellipse([180, 60, 310, 180], fill=(80, 140, 70))

    # Cherry blossoms (dots)
    import random
    random.seed(42)
    for _ in range(40):
        x = random.randint(350, 480)
        y = random.randint(50, 180)
        r = random.randint(4, 8)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 180, 200))

    # Blur for painterly feel
    img = img.filter(ImageFilter.GaussianBlur(radius=2))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def main():
    from test_helpers import get_gemini_client
    try:
        client = get_gemini_client()
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return False

    print("CHECK 3: Veo Video Generation")
    print("=" * 50)
    print("Model: veo-2.0-generate-001")
    print("WARNING: This takes 60-180s and costs ~$2.40")
    print()

    img_bytes = make_stylized_test_image()

    # Save the input image
    with open("test_output/veo_input.png", "wb") as f:
        f.write(img_bytes)
    print(f"Saved input image: test_output/veo_input.png ({len(img_bytes)} bytes)")

    # Use a simple landscape prompt to avoid RAI content filters
    from google.genai import types

    print("Generating 8s video (polling every 5s)...")
    start = time.time()
    try:
        operation = await client.aio.models.generate_videos(
            model="veo-2.0-generate-001",
            prompt="A gentle slow pan across a peaceful landscape painting with warm golden light, trees swaying softly in the breeze, watercolor art style",
            image=types.Image(
                image_bytes=img_bytes,
                mime_type="image/png",
            ),
            config=types.GenerateVideosConfig(
                number_of_videos=1,
                duration_seconds=8,
            ),
        )

        max_wait = 300
        while not operation.done and (time.time() - start) < max_wait:
            elapsed_so_far = int(time.time() - start)
            print(f"  Polling... {elapsed_so_far}s elapsed")
            await asyncio.sleep(5)
            try:
                operation = await client.aio.operations.get(operation)
            except Exception as poll_err:
                print(f"  Polling error (retrying): {poll_err}")
                await asyncio.sleep(3)

        if not operation.done:
            print(f"FAIL: Timed out after {max_wait}s")
            return False

        if operation.result and operation.result.generated_videos:
            video = operation.result.generated_videos[0]
            # Vertex AI returns video_bytes directly; API key mode uses files.download
            if hasattr(video.video, 'video_bytes') and video.video.video_bytes:
                video_data = video.video.video_bytes
            elif hasattr(video.video, 'uri') and video.video.uri:
                video_data = await client.aio.files.download(file=video.video)
            else:
                print(f"FAIL: No video_bytes or uri on video object: {video.video}")
                return False
            result = {"data": video_data}
        else:
            rai = getattr(operation.result, 'rai_media_filtered_count', 0)
            print(f"FAIL: No video returned. RAI filtered: {rai}")
            if hasattr(operation.result, 'rai_media_filtered_reasons'):
                print(f"  Reasons: {operation.result.rai_media_filtered_reasons}")
            return False
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return False
    elapsed = time.time() - start

    if not result:
        print(f"FAIL: No video returned after {elapsed:.1f}s")
        print("Possible causes:")
        print("  - Veo not enabled for your API key (needs billing)")
        print("  - Rate limited")
        print("  - Image format issue")
        return False

    video_data = result["data"]
    if isinstance(video_data, bytes):
        video_path = "test_output/memory_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_data)
        print(f"OK: Video generated in {elapsed:.1f}s ({len(video_data)} bytes)")
        print(f"Saved: {video_path}")
    else:
        print(f"WARN: Video data is {type(video_data)}, not bytes")
        return False

    print()
    print("PASS: Veo video generation working")
    return True


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
