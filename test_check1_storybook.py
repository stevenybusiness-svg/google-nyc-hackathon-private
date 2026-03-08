#!/usr/bin/env python3
"""
CHECK 1: Gemini Interleaved Output (Storybook)
- Sends synthetic screenshots + text prompt to gemini-2.5-flash-image
- Expects interleaved text + generated images back
- Saves: test_output/storybook.html, test_output/storybook_page_*.png
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
from storybook_generator import generate_storybook, render_storybook_html


def make_synthetic_screenshot(color, label):
    """Create a simple synthetic JPEG screenshot with a label."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (320, 240), color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 100), label, fill="white", font=font)
    draw.text((20, 140), "Synthetic test image", fill=(200, 200, 200), font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


async def main():
    from test_helpers import get_gemini_client
    try:
        client = get_gemini_client()
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return False

    # Create 3 synthetic screenshots
    screenshots = [
        {"data": make_synthetic_screenshot((70, 50, 120), "Person A - Kitchen"), "timestamp": "0:15"},
        {"data": make_synthetic_screenshot((50, 100, 70), "Person B - Park"), "timestamp": "1:30"},
        {"data": make_synthetic_screenshot((120, 80, 50), "Both laughing"), "timestamp": "3:00"},
    ]

    storybook_input = {
        "screenshots": screenshots,
        "voice_snippets": [
            {"text": "I made your favorite dumplings today!", "speaker": "Mom (Mandarin)", "timestamp": "0:20"},
            {"text": "I miss you so much, show me!", "speaker": "Daughter (English)", "timestamp": "0:45"},
            {"text": "The park near our house has cherry blossoms now", "speaker": "Mom (Mandarin)", "timestamp": "2:10"},
        ],
        "scene_descriptions": [
            {"timestamp": "0:15", "description": "A warm kitchen with steam rising from a pot, wooden cabinets"},
            {"timestamp": "1:30", "description": "A sunny park with cherry blossom trees in full bloom"},
        ],
        "call_metadata": {
            "duration": "3 minutes",
            "participants": ["Mom (Mandarin)", "Daughter (English)"],
        },
    }

    print("CHECK 1: Gemini Interleaved Output (Storybook)")
    print("=" * 50)
    print("Sending 3 screenshots + voice snippets + scene descriptions...")
    print("Model: gemini-2.5-flash-image")
    print()

    start = time.time()
    pages = await generate_storybook(client, storybook_input)
    elapsed = time.time() - start

    if not pages:
        print(f"FAIL: No pages returned after {elapsed:.1f}s")
        return False

    text_count = sum(1 for p in pages if p["type"] == "text")
    image_count = sum(1 for p in pages if p["type"] == "image")

    print(f"OK: Got {len(pages)} pages ({text_count} text, {image_count} images) in {elapsed:.1f}s")

    # Save individual images
    for i, page in enumerate(pages):
        if page["type"] == "image":
            img_data = page["content"]
            if isinstance(img_data, str):
                img_data = base64.b64decode(img_data)
            path = f"test_output/storybook_page_{i}.png"
            with open(path, "wb") as f:
                f.write(img_data)
            print(f"  Saved: {path} ({len(img_data)} bytes)")

    # Render and save HTML
    html = render_storybook_html(pages, title="Test Storybook - Mom & Daughter")
    html_path = "test_output/storybook.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  Saved: {html_path}")

    print()
    print(f"PASS: Storybook generated successfully")
    return True


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
