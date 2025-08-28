import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GENAI_API_KEY")
model = os.getenv("GEMINI_MODEL") or "gemini-2.0-pro-latest"
api_version = os.getenv("GEMINI_API_VERSION") or "v1beta"
api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent"

def stop_sequence_demo():
    prompt_text = "List three fruits:"
    payload = {
        "model": model,
        "contents": [{
            "parts": [{
                "text": prompt_text
            }]
        }],
        "generationConfig": {
            "stopSequences": ["\n"],
            "maxOutputTokens": 50
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
        output_text = data['candidates'][0]['content']['parts'][0]['text']
        print("Output with stop sequence:")
        print(output_text.strip())

    except requests.exceptions.RequestException as e:
        print(f"RequestException: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    stop_sequence_demo()