import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv("GENAI_API_KEY")
model = os.getenv("GEMINI_MODEL") or "gemini-2.0-pro-latest"
api_version = os.getenv("GEMINI_API_VERSION") or "v1beta"
api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent"

def one_shot_prompting(user_query, example_input, example_output):
    """
    Demonstrates one-shot prompting with the Gemini API.

    Args:
        user_query (str): The user's query.
        example_input (str): An example input.
        example_output (str): The corresponding example output.

    Returns:
        str: The generated response from the Gemini API.
    """

    prompt_text = f"""
    Here's an example:
    Input: {example_input}
    Output: {example_output}

    Now, answer this:
    {user_query}
    """

    payload = {
        "model": model,
        "contents": [{
            "parts": [{
                "text": prompt_text
            }]
        }],
        "generationConfig": {
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
        return output_text.strip()

    except requests.exceptions.RequestException as e:
        return f"RequestException: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    user_query = "What are three popular tourist attractions in London?"
    example_input = "What are three popular tourist attractions in Paris?"
    example_output = "Eiffel Tower, Louvre Museum, Arc de Triomphe"

    response = one_shot_prompting(user_query, example_input, example_output)
    print(f"User Query: {user_query}")
    print(f"Gemini Response: {response}")