import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv("GENAI_API_KEY")
model = os.getenv("GEMINI_MODEL") or "gemini-2.0-pro-latest"
api_version = os.getenv("GEMINI_API_VERSION") or "v1beta"

def generate_embeddings(text):
    """Generates embeddings for the given text using the Gemini API."""
    api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:embedContent"

    payload = {
        "model": model,
        "content": {
            "parts": [{"text": text}]
        }
    }

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        embedding = data["embedding"]["values"]
        return embedding
    except requests.exceptions.RequestException as e:
        print(f"RequestException: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    text_to_embed = "This is a sample text for generating embeddings."
    embeddings = generate_embeddings(text_to_embed)

    if embeddings:
        print("Embeddings generated successfully:")
        print(embeddings)
    else:
        print("Failed to generate embeddings.")