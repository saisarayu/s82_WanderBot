# s82_WanderBot

WanderBot is an interactive chatbot powered by Google's Gemini API.  
It supports dynamic prompting, allowing users to interact with the bot and receive helpful, concise responses.

## Features

- Zero-shot chat interface using Gemini API
- Dynamic system and user prompts
- Easy configuration via `.env` file

## Getting Started

1. Install dependencies:
   ```
   pip install requests python-dotenv
   ```
2. Set up your `.env` file with:
   ```
   GENAI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-2.0
   GEMINI_API_VERSION=v1beta
   ```
3. Run the bot:
   ```
   python ZeroShot.py
   ```

## Dynamic Prompting

The bot uses dynamic prompting to adapt its responses based on user input and system instructions.  
You can modify the system instructions in `ZeroShot.py