import base64
import logging
from google import genai
from google.genai.types import GenerateContentConfig, Modality

logger = logging.getLogger(__name__)

STORYBOOK_PROMPT = """
Create an interactive storybook capturing this video call between two loved ones separated by distance.

CALL DATA:
- Duration: {duration}
- Participants: {participants}

KEY VOICE MOMENTS:
{voice_snippets}

SCENE DESCRIPTIONS:
{scene_descriptions}

INSTRUCTIONS:
Generate a flowing narrative that weaves together:
1. Text telling the story of their time together
2. Generated images (stylized, warm, memory-like) for each key moment
3. The voice snippets as quoted moments (in italics)
4. The feeling of presence despite distance

STYLE:
- Warm, intimate, like a handwritten letter with photos
- Each "page" should have text followed by a generated image
- Highlight emotional moments: when they laughed, when they said "I love you"
- End with a reflection on connection despite distance

FORMAT:
Generate interleaved text and images. Each section should be:
[Text describing the moment]
[Generated image capturing that moment]
[Text continuing the narrative]
...

Do NOT use literal screenshots. Generate NEW images in a warm, stylized, memory-like aesthetic.
"""

async def generate_storybook(client: genai.Client, storybook_input: dict) -> list:
    """Generate an interactive storybook from call data."""
    try:
        voice_snippets = storybook_input.get("voice_snippets", [])
        scene_descs = storybook_input.get("scene_descriptions", [])
        screenshots = storybook_input.get("screenshots", [])[:5]
        call_meta = storybook_input.get("call_metadata", {})

        voice_text = "\n".join([
            f"- \"{s.get('text', '')}\" ({s.get('speaker', '?')}, {s.get('timestamp', '?')})"
            for s in voice_snippets if isinstance(s, dict)
        ]) or "None captured."

        scene_text = "\n".join([
            f"- {s.get('timestamp', '?')}: {s.get('description', '')}"
            for s in scene_descs if isinstance(s, dict)
        ]) or "None captured."

        prompt = STORYBOOK_PROMPT.format(
            duration=call_meta.get("duration", "Unknown"),
            participants=", ".join(call_meta.get("participants", [])),
            voice_snippets=voice_text,
            scene_descriptions=scene_text,
        )

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
                contents.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_data,
                    }
                })
        contents.append({"text": prompt})

        has_images = any("inline_data" in c for c in contents)
        modalities = [Modality.TEXT, Modality.IMAGE] if has_images else [Modality.TEXT]

        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=GenerateContentConfig(
                response_modalities=modalities,
            ),
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
                    pages.append({
                        "type": "image",
                        "content": img,
                        "mime_type": part.inline_data.mime_type or "image/png",
                    })

        if not pages:
            logger.warning("Storybook generation returned no content")
        return pages
    except Exception as e:
        logger.error(f"Error generating storybook: {type(e).__name__}: {e}")
        return []

def render_storybook_html(pages: list, title: str = "Our Moment Together") -> str:
    """Render storybook pages to HTML."""
    html_parts = [f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Georgia', serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 40px;
                background: #faf8f5;
                color: #333;
            }}
            h1 {{
                text-align: center;
                font-weight: normal;
                color: #5a4a3a;
            }}
            .page {{
                margin: 40px 0;
            }}
            .page-text {{
                font-size: 18px;
                line-height: 1.8;
                margin: 20px 0;
            }}
            .page-image {{
                width: 100%;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }}
            .quote {{
                font-style: italic;
                color: #7a6a5a;
                border-left: 3px solid #d4c4b4;
                padding-left: 20px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
    """]
    
    for page in pages:
        if page["type"] == "text":
            html_parts.append(f'<div class="page-text">{page["content"]}</div>')
        elif page["type"] == "image":
            raw = page["content"]
            if isinstance(raw, bytes):
                img_data = base64.b64encode(raw).decode()
            elif isinstance(raw, str):
                img_data = raw
            else:
                continue
            mime = page.get("mime_type", "image/png")
            html_parts.append(
                f'<img class="page-image" src="data:{mime};base64,{img_data}">'
            )
            
    html_parts.append("</body></html>")
    return "\n".join(html_parts)
