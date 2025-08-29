import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv("GENAI_API_KEY")
model = os.getenv("GEMINI_MODEL") or "gemini-2.0-pro-latest"
api_version = os.getenv("GEMINI_API_VERSION") or "v1beta"
api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent"

def parse_travel_submission(submission_text):
    system_prompt = """
    You are WanderBot, an assistant that converts travel experiences into structured JSON
    with the following fields: name, location, description, season, and recommended_activity.
    Always respond in valid JSON only.
    """

    user_prompt = f"""
    Here is a new travel submission:
    {submission_text}
    Please convert this submission into structured JSON.
    """

    payload = {
        "model": model,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            },
            {
                "role": "model",
                "parts": [{"text": system_prompt}]
            }
        ],
        "generationConfig": {
            "stopSequences": ["\n\n"],
            "maxOutputTokens": 200
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

        try:
            parsed_data = json.loads(output_text)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print(f"Raw Output from Gemini: {output_text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"RequestException: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    submission = """
    Name: John Doe
    Location: Paris
    Experience: I visited the Eiffel Tower and loved the view. It was a sunny day.
    """
    parsed_data = parse_travel_submission(submission)
    if parsed_data:
        print(parsed_data)