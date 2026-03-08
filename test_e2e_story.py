#!/usr/bin/env python3
"""
End-to-end test: Transcript → Story Scenes → Storybook → Video

Simulates: Grandma tells a story during a video call,
we extract the story, generate an illustrated storybook,
then create a memory video from the storybook images.
"""
import asyncio
import base64
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

# Sample transcript — grandma telling a bedtime story during a call
SAMPLE_TRANSCRIPT = """
Grandma: Hello my darling! How are you today?
Child: Hi grandma! I'm good! Tell me a story please!
Grandma: Oh you want a story? Okay, let me tell you about the little cloud who was afraid of thunder.
Child: Ooh yes!
Grandma: Once upon a time, high up in the sky, there lived a tiny little cloud named Nimbus.
Nimbus was the smallest cloud in the whole sky, and he was soft and fluffy like cotton candy.
Child: Like cotton candy! I love cotton candy!
Grandma: Yes, just like cotton candy! Now, Nimbus loved floating around with his cloud friends.
They would drift over mountains and rivers, making funny shapes for the children below to see.
But there was one thing that scared little Nimbus very much — thunder!
Every time a storm came, the big clouds would rumble and flash with lightning,
and poor little Nimbus would hide behind the tallest mountain he could find.
Child: Aww, poor Nimbus!
Grandma: One day, the biggest storm of the year was coming. All the clouds gathered together
because they needed every cloud to help make the rain. The flowers and trees were so thirsty!
But Nimbus was too scared. He floated away and hid behind old Mountain Maple.
Then a little sunflower down below called up to him — "Please little cloud! We need your rain!
I'm so thirsty and my petals are drooping!"
Child: Oh no! Did Nimbus help?
Grandma: Nimbus looked down and saw the little sunflower and all the other flowers wilting.
He took a deep breath — well, as deep as a cloud can breathe — and he floated back to join the others.
When the thunder rumbled, he was scared, but he thought of the little sunflower.
And do you know what? When Nimbus let his rain fall, the most beautiful rainbow appeared!
The sunflower looked up and smiled, and Nimbus realized that the thunder wasn't so scary after all —
it was just the sky's way of cheering for the rain!
Child: I love that story grandma! Nimbus is brave!
Grandma: Yes he is, just like you my darling. Now you go to sleep and dream of rainbows, okay?
Child: Okay grandma, I love you!
Grandma: I love you too, sweet dreams.
"""


async def main():
    from test_helpers import get_gemini_client
    from storybook_generator import extract_story_scenes, generate_storybook, render_storybook_html, get_storybook_images, generate_all_narrations
    from memory_video import pair_scenes_with_media, create_story_video_hybrid

    try:
        client = get_gemini_client()
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return False

    print("=" * 60)
    print("END-TO-END: Transcript → Storybook → Video")
    print("=" * 60)

    # ─── Step 1: Extract story scenes ───
    print("\n[Step 1] Extracting story scenes from transcript...")
    start = time.time()
    scenes = await extract_story_scenes(client, SAMPLE_TRANSCRIPT, storyteller="Grandma")
    elapsed = time.time() - start

    if not scenes:
        print(f"FAIL: No scenes extracted after {elapsed:.1f}s")
        return False

    print(f"OK: {len(scenes)} scenes extracted in {elapsed:.1f}s")
    for i, scene in enumerate(scenes, 1):
        print(f"  Scene {i}: {scene.get('title', '?')}")
        print(f"    Story: {scene.get('story_text', '?')[:80]}...")

    # ─── Step 2: Generate illustrated storybook ───
    print(f"\n[Step 2] Generating illustrated storybook ({len(scenes)} scenes)...")
    start = time.time()
    pages = await generate_storybook(client, {
        "scenes": scenes,
        "storyteller": "Grandma",
        "title": "The Little Cloud Who Was Afraid of Thunder",
    })
    elapsed = time.time() - start

    text_count = sum(1 for p in pages if p["type"] == "text")
    image_count = sum(1 for p in pages if p["type"] == "image")

    if not pages:
        print(f"FAIL: No pages generated after {elapsed:.1f}s")
        return False

    print(f"OK: {len(pages)} pages ({text_count} text, {image_count} images) in {elapsed:.1f}s")

    # ─── Step 2b: Generate audio narration ───
    print(f"\n[Step 2b] Generating audio narration for {text_count} text pages...")
    start = time.time()
    narrations = await generate_all_narrations(client, pages, voice_name="Kore")
    elapsed = time.time() - start
    success_count = sum(1 for n in narrations if n is not None)
    print(f"OK: {success_count}/{text_count} narrations generated in {elapsed:.1f}s")

    # Save individual audio files
    for i, narration in enumerate(narrations):
        if narration:
            audio_data = narration["data"]
            if isinstance(audio_data, bytes):
                ext = "wav" if "wav" in narration.get("mime_type", "") else "mp3"
                path = f"test_output/e2e_narration_{i}.{ext}"
                with open(path, "wb") as f:
                    f.write(audio_data)
                print(f"  Saved: {path} ({len(audio_data)} bytes)")

    # Save storybook with audio
    html = render_storybook_html(pages, title="The Little Cloud Who Was Afraid of Thunder", audio_narrations=narrations)
    with open("test_output/e2e_storybook.html", "w") as f:
        f.write(html)
    print(f"  Saved: test_output/e2e_storybook.html (with audio players)")

    # Save individual images
    for i, page in enumerate(pages):
        if page["type"] == "image":
            img_data = page["content"]
            if isinstance(img_data, str):
                img_data = base64.b64decode(img_data)
            path = f"test_output/e2e_page_{i}.png"
            with open(path, "wb") as f:
                f.write(img_data)
            print(f"  Saved: {path} ({len(img_data)} bytes)")

    # ─── Step 3: Create hybrid narrated story video (Veo + Ken Burns) ───
    scenes_with_media = pair_scenes_with_media(pages, narrations, scenes)
    if not scenes_with_media:
        print("\nWARN: No paired scenes, skipping video")
        return True

    dramatic_count = sum(1 for s in scenes_with_media if s.get("dramatic"))
    print(f"\n[Step 3] Creating hybrid story video ({len(scenes_with_media)} scenes, {dramatic_count} dramatic/Veo)...")
    for i, s in enumerate(scenes_with_media):
        mode = "VEO" if s.get("dramatic") else "Ken Burns"
        print(f"  Scene {i}: '{s['title']}' [{mode}] — {len(s['images'])} images")

    start = time.time()
    video_result = await create_story_video_hybrid(client, scenes_with_media)
    elapsed = time.time() - start

    if video_result:
        video_data = video_result["data"]
        with open("test_output/e2e_story_video.mp4", "wb") as f:
            f.write(video_data)
        print(f"OK: Hybrid story video created in {elapsed:.1f}s "
              f"({video_result['duration_seconds']:.1f}s total, {len(video_data)} bytes)")
        print(f"  Saved: test_output/e2e_story_video.mp4")
    else:
        print(f"WARN: Story video creation failed after {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("PASS: End-to-end pipeline complete!")
    print("=" * 60)
    print("\nOutputs:")
    print("  test_output/e2e_storybook.html   — open in browser (with audio)")
    print("  test_output/e2e_page_*.png       — storybook illustrations")
    print("  test_output/e2e_story_video.mp4  — hybrid narrated story video")
    return True


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    os.makedirs("test_output", exist_ok=True)
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
