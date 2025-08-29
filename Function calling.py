import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GENAI_API_KEY")
model = os.getenv("GEMINI_MODEL") or "gemini-2.0-pro-latest"
api_version = os.getenv("GEMINI_API_VERSION") or "v1beta"
api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent"

def get_current_weather(location):
    """
    Gets the current weather for a given location.

    Args:
        location (str): The city and state, e.g., "San Francisco, CA"

    Returns:
        dict: A dictionary containing the weather information, or None if an error occurs.
    """
    # In a real application, you would call a weather API here.
    # For this example, we'll just return a dummy response.
    if location.lower() == "san francisco, ca":
        return {"location": "San Francisco, CA", "temperature": "72", "forecast": "Sunny with a chance of fog"}
    elif location.lower() == "london, uk":
        return {"location": "London, UK", "temperature": "60", "forecast": "Cloudy with a chance of rain"}
    else:
        return None

def call_gemini_with_function_calling(user_query):
    """
    Calls the Gemini API with function calling to get the weather for a location.

    Args:
        user_query (str): The user's query, e.g., "What's the weather like in San Francisco?"

    Returns:
        str: The response from the Gemini API, which may include a function call.
    """

    # Define the function to be called
    function_description = {
        "name": "get_current_weather",
        "description": "Gets the current weather for a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA",
                },
            },
            "required": ["location"],
        },
    }

    payload = {
        "model": model,
        "contents": [{
            "parts": [{
                "text": user_query
            }]
        }],
        "tools": [{"function_declarations": [function_description]}],
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

        # Check if the response contains a function call
        if "tool_calls" in data['candidates'][0]['content']:
            tool_calls = data['candidates'][0]['content']['tool_calls']
            function_name = tool_calls[0]['function']['name']
            arguments = json.loads(tool_calls[0]['function']['arguments'])

            if function_name == "get_current_weather":
                weather_info = get_current_weather(arguments["location"])
                if weather_info:
                    return f"The weather in {weather_info['location']} is {weather_info['temperature']} and {weather_info['forecast']}"
                else:
                    return "Sorry, I couldn't retrieve the weather for that location."
            else:
                return "I don't know how to call that function."
        else:
            # If no function call, return the text response
            output_text = data['candidates'][0]['content']['parts'][0]['text']
            return output_text.strip()

    except requests.exceptions.RequestException as e:
        return f"RequestException: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    user_query = "What's the weather like in San Francisco, CA?"
    response = call_gemini_with_function_calling(user_query)
    print(f"User Query: {user_query}")
    print(f"Gemini Response: {response}")

    user_query_london = "What's the weather like in London, UK?"
    response_london = call_gemini_with_function_calling(user_query_london)
    print(f"User Query: {user_query_london}")
    print(f"Gemini Response: {response_london}")