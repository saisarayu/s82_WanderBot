import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GENAI_API_KEY")
MODEL = "gemini-2.0"

url = f"https://api.generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

payload = {
    "prompt": "Explain quantum physics in simple terms", # <prompt> is here
    "temperature": 0.7,  # Update the temperature here
    "maxOutputTokens": 200
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())