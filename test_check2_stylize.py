#!/usr/bin/env python3
"""
CHECK 2: Gemini Image Stylization
- Sends a synthetic screenshot to gemini-2.5-flash-image with stylization prompt
- Expects a warm, watercolor-style image back
- Saves: test_output/original.jpg, test_output/stylized_0.png, stylized_1.png
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
from memory_video import stylize_image, stylize_all_images


def make_synthetic_photo(scene="kitchen"):
    """Create a more detailed synthetic photo for stylization testing."""
    from PIL import Image, ImageDraw, ImageFont

    if scene == "kitchen":
        bg = (180, 140, 100)
        elements = [
            ((50, 50, 200, 180), (140, 100, 60)),    # table
            ((220, 30, 280, 120), (80, 80, 90)),      # window
            ((100, 60, 150, 110), (200, 50, 30)),     # pot
            ((80, 120, 160, 170), (220, 200, 180)),   # plate
        ]
        label = "Warm Kitchen Scene"
    else:
        bg = (100, 160, 100)
        elements = [
            ((0, 160, 320, 240), (80, 130, 60)),      # grass
            ((50, 20, 90, 140), (100, 70, 40)),        # tree trunk
            ((20, 0, 120, 60), (60, 140, 60)),         # foliage
            ((200, 60, 280, 160), (255, 180, 200)),    # cherry blossoms
        ]
        label = "Park with Cherry Blossoms"

    img = Image.new("RGB", (320, 240), bg)
    draw = ImageDraw.Draw(img)
    for rect, color in elements:
        draw.rectangle(rect, fill=color)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 210), label, fill="white", font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), base64.b64encode(buf.getvalue()).decode()


async def main():
    from test_helpers import get_gemini_client
    try:
        client = get_gemini_client()
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return False

    print("CHECK 2: Gemini Image Stylization")
    print("=" * 50)
    print("Model: gemini-2.5-flash-image")
    print()

    # --- Single image test ---
    print("Test 2a: Single image stylization")
    raw_bytes, b64_data = make_synthetic_photo("kitchen")

    # Save original for comparison
    with open("test_output/original_kitchen.jpg", "wb") as f:
        f.write(raw_bytes)
    print(f"  Saved original: test_output/original_kitchen.jpg ({len(raw_bytes)} bytes)")

    start = time.time()
    result = await stylize_image(client, b64_data)
    elapsed = time.time() - start

    if not result:
        print(f"  FAIL: No stylized image returned after {elapsed:.1f}s")
        return False

    img_data = result["data"]
    if isinstance(img_data, str):
        img_data = base64.b64decode(img_data)
    with open("test_output/stylized_0.png", "wb") as f:
        f.write(img_data)
    print(f"  OK: Stylized image returned in {elapsed:.1f}s ({len(img_data)} bytes)")
    print(f"  Saved: test_output/stylized_0.png")

    # --- Batch test (2 images concurrently) ---
    print()
    print("Test 2b: Batch stylization (2 images concurrent)")
    screenshots = [
        {"data": make_synthetic_photo("kitchen")[1]},
        {"data": make_synthetic_photo("park")[1]},
    ]

    # Save second original
    raw2, _ = make_synthetic_photo("park")
    with open("test_output/original_park.jpg", "wb") as f:
        f.write(raw2)

    start = time.time()
    results = await stylize_all_images(client, screenshots, max_images=2)
    elapsed = time.time() - start

    if not results:
        print(f"  FAIL: No images returned after {elapsed:.1f}s")
        return False

    for i, r in enumerate(results):
        img_data = r["data"]
        if isinstance(img_data, str):
            img_data = base64.b64decode(img_data)
        path = f"test_output/stylized_batch_{i}.png"
        with open(path, "wb") as f:
            f.write(img_data)
        print(f"  Saved: {path} ({len(img_data)} bytes)")

    print(f"  OK: {len(results)} stylized images in {elapsed:.1f}s")
    print()
    print("PASS: Image stylization working")
    return True


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
