"""Shared helper for test scripts — creates Gemini client from .env"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def get_gemini_client() -> genai.Client:
    """Create Gemini client using env vars.

    Supports:
      - GOOGLE_GENAI_USE_VERTEXAI=True + GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION (Vertex AI)
      - GOOGLE_API_KEY (API key mode)
    """
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    api_key = os.getenv("GOOGLE_API_KEY")

    if use_vertex and project:
        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
        print(f"Using Vertex AI (project={project}, location={location})")
        return client
    elif api_key:
        client = genai.Client(api_key=api_key)
        print("Using API key mode")
        return client
    else:
        raise RuntimeError("Set GOOGLE_GENAI_USE_VERTEXAI+GOOGLE_CLOUD_PROJECT or GOOGLE_API_KEY in .env")
