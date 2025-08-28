import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GENAI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
API_VERSION = os.getenv("GEMINI_API_VERSION", "v1beta")

def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/{API_VERSION}/models/{MODEL}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": API_KEY}
    data = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7}
    }
    
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    result = resp.json()

    # ✅ Log tokens if usageMetadata is present
    usage = result.get("usageMetadata", {})
    if usage:
        print(f"🔹 Tokens used: prompt={usage.get('promptTokenCount', 0)}, "
              f"completion={usage.get('candidatesTokenCount', 0)}, "
              f"total={usage.get('totalTokenCount', 0)}")

    return result["candidates"][0]["content"]["parts"][0]["text"]

if __name__ == "__main__":
    reply = call_gemini("Give me a 2-day trip plan for Munnar.")
    print("🤖:", reply)
