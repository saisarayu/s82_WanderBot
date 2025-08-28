import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GENAI_API_KEY")  # Changed to GENAI_API_KEY
model = os.getenv("GEMINI_MODEL") or "gemini-2.0-pro-latest"
api_version = os.getenv("GEMINI_API_VERSION") or "v1beta"
api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent"

try:
    payload = {
        "model": model,
        "contents": [{
            "parts": [{
                "text": "Give me user info: name, age, city for John, 25, New York."
            }]
        }],
        "generationConfig": {
            "temperature": 0,
        },
        "tools": []
    }

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }

    response = requests.post(api_url, json=payload, headers=headers)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    data = response.json()

    # Extract the text response
    output_text = data['candidates'][0]['content']['parts'][0]['text']

    # Try to parse the JSON
    try:
        user_info = json.loads(output_text)
        print(user_info['name'])
        print(user_info['age'])
        print(user_info['city'])

    except (json.JSONDecodeError, KeyError) as e:
        print(f"JSONDecodeError or KeyError: {e}")
        print(f"Raw Output from Gemini: {output_text}")

except requests.exceptions.RequestException as e:
    print(f"RequestException: {e}")
except Exception as e:
    print(f"An error occurred: {e}")