import os
import requests
from dotenv import load_dotenv

def load_api_config():
    load_dotenv()
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("GENAI_API_KEY environment variable not set.")
    model = os.getenv("GEMINI_MODEL") or "gemini-2.0"  # use gemini-2.0 for v1beta
    api_version = os.getenv("GEMINI_API_VERSION") or "v1beta"
    api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent"
    return api_url, api_key, model

def chat():
    api_url, api_key, model = load_api_config()
    print("✅ ZeroShot.py is running inside the virtual environment!")
    print("Start chatting with Gemini! Type 'exit' to quit.")

    system_instructions = "You are a helpful assistant that responds in a friendly and concise style."

    try:
        while True:
            user_input = input("User: ")
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            payload = {
                "model": model,
                "temperature": 0.7,
                "candidate_count": 1,
                "prompt": {
                    "messages": [
                        {"author": "system", "content": [{"type": "text", "text": system_instructions}]},
                        {"author": "user", "content": [{"type": "text", "text": user_input}]}
                    ]
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
                reply = data["candidates"][0]["content"][0]["text"]
                print("Gemini:", reply)
            except Exception as e:
                print("Error communicating with Gemini:", e)

    except KeyboardInterrupt:
        print("\nGoodbye! (interrupted)")

if __name__ == "__main__":
    chat()
