"""
WanderBot – Dynamic Prompting Module
------------------------------------

Drop-in dynamic prompting for your travel assistant. Designed to work with
Google Generative Language (Gemini) via REST (requests) and a simple
in-memory conversation store.

Environment vars expected (same style as your ZeroShot.py):
- GENAI_API_KEY
- GEMINI_MODEL (default: gemini-2.0-flash)
- GEMINI_API_VERSION (default: v1beta)

Usage (quickstart):

    from wanderbot_dynamic_prompt import WanderBot

    bot = WanderBot(
        bot_name="WanderBot",
        app_purpose="Help users plan trips, find hotels, food, and tips based on season and weather.",
        user_profile={"name": "Sarayu", "home": "Madurai", "preferences": ["budget", "vegetarian"]},
        tools=[
            {
                "name": "get_hotels",
                "description": "Return hotel options for a city with budget filters.",
                "schema": {
                    "city": "str",
                    "check_in": "YYYY-MM-DD",
                    "check_out": "YYYY-MM-DD",
                    "budget_per_night_inr": "int"
                }
            },
            {
                "name": "get_weather",
                "description": "Return weather for a city and date.",
                "schema": {"city": "str", "date": "YYYY-MM-DD"}
            }
        ]
    )

    answer = bot.respond("3-day monsoon-friendly itinerary for Munnar under ₹5k?")
    print(answer)

You can also plug your own tool-calling layer: implement the stubs in
ToolRouter.call_tool and feed results back into bot.respond(..., tool_result=...).
"""
from __future__ import annotations

import os
import re
import json
import time
import math
import uuid
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import requests

# ------------------------------
# Utility: season detection
# ------------------------------

def infer_season_from_month(month: int) -> str:
    # Adjust if you want country-specific mapping. This is a simple India-leaning map.
    if month in (6, 7, 8, 9):
        return "monsoon"
    if month in (10, 11):
        return "post-monsoon"
    if month in (12, 1, 2):
        return "winter"
    return "summer"

# ------------------------------
# Safety + PII
# ------------------------------

PII_PATTERNS = [
    re.compile(r"\b\d{12}\b"),  # Aadhaar-like 12 digits
    re.compile(r"\b\d{16}\b"),  # card numbers (very naive)
    re.compile(r"\b[0-9]{10}\b"),  # phone
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # email
]


def redact_pii(text: str) -> str:
    redacted = text
    for pat in PII_PATTERNS:
        redacted = pat.sub("[REDACTED]", redacted)
    return redacted


# ------------------------------
# Few-shot Library
# ------------------------------

FEW_SHOTS: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": "find me cheap hotels in goa for this weekend",
    },
    {
        "role": "assistant",
        "content": (
            "Intent: hotel_search\n"
            "Entities: city=Goa, dates=this weekend, constraints=cheap\n"
            "Plan: 1) check weather 2) fetch hotels 3) sort by price 4) suggest transport\n"
            "Answer (friendly + concise): Here are budget-friendly stays in Goa for the weekend..."
        ),
    },
    {
        "role": "user",
        "content": "itinerary for 2 days in Pondicherry, vegetarian food only",
    },
    {
        "role": "assistant",
        "content": (
            "Intent: itinerary\n"
            "Entities: city=Pondicherry, duration=2 days, diet=vegetarian\n"
            "Plan: morning/afternoon/evening blocks with veg eateries near sights\n"
            "Answer: Day 1: Promenade Beach sunrise... veg cafés nearby... Day 2: Auroville..."
        ),
    },
]

# ------------------------------
# Conversation Memory
# ------------------------------

class ConversationMemory:
    def __init__(self, max_tokens: int = 6000, target_context_tokens: int = 3000):
        self.messages: List[Dict[str, str]] = []
        self.max_tokens = max_tokens
        self.target_context_tokens = target_context_tokens

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._shrink_if_needed()

    def summary(self, max_len: int = 800) -> str:
        # Very simple heuristic summary: keep last N turns + compress earlier.
        if not self.messages:
            return ""
        last = self.messages[-6:]
        earlier = self.messages[:-6]
        earlier_text = " ".join(m["content"] for m in earlier)
        earlier_hint = (earlier_text[: max(0, max_len - 40)] + "…") if earlier else ""
        last_text = "\n".join(f"{m['role']}: {m['content']}" for m in last)
        return f"Earlier gist: {earlier_hint}\nRecent turns:\n{last_text}"

    def _approx_tokens(self, text: str) -> int:
        # Rough token estimate ~ 4 chars per token
        return max(1, math.ceil(len(text) / 4))

    def _shrink_if_needed(self) -> None:
        total = sum(self._approx_tokens(m["content"]) for m in self.messages)
        if total <= self.max_tokens:
            return
        # Drop from the oldest until under target
        kept: List[Dict[str, str]] = []
        running = 0
        for m in reversed(self.messages):
            t = self._approx_tokens(m["content"]) 
            if running + t > self.target_context_tokens:
                break
            kept.append(m)
            running += t
        self.messages = list(reversed(kept))


# ------------------------------
# Context Builder
# ------------------------------

class ContextBuilder:
    def __init__(self, app_purpose: str, user_profile: Optional[Dict[str, Any]] = None):
        self.app_purpose = app_purpose
        self.user_profile = user_profile or {}

    def build(self, user_message: str, hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        now = dt.datetime.now()
        season = infer_season_from_month(now.month)
        ctx = {
            "timestamp": now.isoformat(),
            "season": season,
            "app_purpose": self.app_purpose,
            "user_profile": self.user_profile,
            "hints": hints or {},
            "sanitized_user_message": redact_pii(user_message.strip()),
        }
        return ctx


# ------------------------------
# Prompt Renderer
# ------------------------------

SYSTEM_PROMPT_TEMPLATE = """
You are {bot_name}, a friendly, practical travel assistant for Indian users.
Goal: {app_purpose}
Style: concise, optimistic, step-by-step when planning. Use Indian context (₹, train/bus options) when relevant.
Safety: Do not request or store sensitive PII. If user shares PII, redact and proceed.
Tool use: If tools are available, explain when you used them and incorporate results.
If the user asks for medical or legal advice, give general info + advise consulting a professional.
""".strip()


def render_tools_block(tools: List[Dict[str, Any]]) -> str:
    if not tools:
        return "(No external tools available)"
    lines = ["Available Tools:"]
    for t in tools:
        schema_json = json.dumps(t.get("schema", {}), ensure_ascii=False)
        lines.append(f"- {t['name']}: {t.get('description','')}. Schema: {schema_json}")
    return "\n".join(lines)


def render_few_shots() -> List[Dict[str, str]]:
    return FEW_SHOTS[:]


def render_prompt(system: str, context: Dict[str, Any], memory_summary: str, tools_block: str, user_message: str, tool_result: Optional[str] = None) -> List[Dict[str, str]]:
    context_block = (
        f"Time: {context['timestamp']} | Season: {context['season']}\n"
        f"User profile: {json.dumps(context['user_profile'], ensure_ascii=False)}\n"
        f"Hints: {json.dumps(context['hints'], ensure_ascii=False)}\n"
    )

    guardrails = (
        "Tasks: Classify intent → extract entities (city, dates, budget, food prefs) → plan → answer.\n"
        "Avoid hallucinating unavailable data; ask for missing key details in 1-2 short questions if essential.\n"
        "Prefer bullet itineraries with morning/afternoon/evening blocks. Use ₹ for currency."
    )

    assistant_preamble = (
        f"[Context]\n{context_block}\n[Memory]\n{memory_summary or 'None'}\n[Tools]\n{tools_block}\n[Guardrails]\n{guardrails}\n"
    )

    msgs: List[Dict[str, str]] = []
    msgs.append({"role": "system", "content": system})
    msgs.extend(render_few_shots())
    msgs.append({"role": "assistant", "content": assistant_preamble})
    msgs.append({"role": "user", "content": user_message})
    if tool_result:
        msgs.append({"role": "assistant", "content": f"Tool result incorporated: {tool_result}"})
    return msgs


# ------------------------------
# Gemini REST Client (minimal)
# ------------------------------

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, api_version: Optional[str] = None):
        self.api_key = api_key or os.getenv("GENAI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL") or "gemini-2.0-flash"
        self.api_version = api_version or os.getenv("GEMINI_API_VERSION") or "v1beta"
        self.base = f"https://generativelanguage.googleapis.com/{self.api_version}/models/{self.model}:generateContent"

    def generate(self, messages: List[Dict[str, str]], fallback_local: bool = True) -> str:
        if not self.api_key:
            if fallback_local:
                return self._local_stub(messages)
            raise EnvironmentError("GENAI_API_KEY not set.")

        # Convert messages into Gemini's content format (role->parts)
        contents = []
        for m in messages:
            contents.append({
                "role": "user" if m["role"] == "user" else "model" if m["role"] == "assistant" else "system",
                "parts": [{"text": m["content"]}]
            })

        payload = {"contents": contents, "generationConfig": {"temperature": 0.6, "top_p": 0.9}}
        params = {"key": self.api_key}
        resp = requests.post(self.base, params=params, json=payload, timeout=60)
        if resp.status_code != 200:
            # Graceful fallback
            return self._local_stub(messages, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            return self._local_stub(messages, error="Malformed Gemini response")

    def _local_stub(self, messages: List[Dict[str, str]], error: Optional[str] = None) -> str:
        # Very basic offline behavior for dev/testing.
        user_last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        apology = f" (Note: Using local stub{' – ' + error if error else ''}.)"
        return (
            "I can't reach Gemini right now, but here's a quick plan:" + apology + "\n\n"
            "• Intent guess: travel_planning\n"
            "• Plan: 1) confirm dates/budget 2) check weather 3) propose itinerary 4) list hotels/food\n"
            f"• Suggestion for your request → {user_last[:120]}...\n"
            "  - Morning: sightseeing\n  - Afternoon: local food\n  - Evening: market walk\n  - Budget tips: use buses/metros, book early"
        )


# ------------------------------
# Tool Router (stub - plug your backends here)
# ------------------------------

class ToolRouter:
    def __init__(self, tools: Optional[List[Dict[str, Any]]] = None):
        self.tools = {t["name"]: t for t in (tools or [])}

    def call_tool(self, name: str, args: Dict[str, Any]) -> str:
        # TODO: Wire this to your real services (hotels API, weather API, etc.)
        if name not in self.tools:
            return f"Tool '{name}' not found."
        # Simulated results
        if name == "get_weather":
            city = args.get("city", "Unknown")
            date = args.get("date", str(dt.date.today()))
            return f"Weather in {city} on {date}: light showers, 24–28°C, humid."
        if name == "get_hotels":
            city = args.get("city", "Unknown")
            budget = args.get("budget_per_night_inr", 2500)
            return (
                f"Top budget stays in {city} under ₹{budget}:\n"
                "- Cozy Inn (₹1800, near bus station)\n"
                "- Lakeside Lodge (₹2200, breakfast included)\n"
                "- City Capsule (₹1200, dorms)"
            )
        return f"Tool '{name}' executed with args {args}, but no mock implemented."


# ------------------------------
# WanderBot – Orchestrator
# ------------------------------

class WanderBot:
    def __init__(self, bot_name: str, app_purpose: str, user_profile: Optional[Dict[str, Any]] = None, tools: Optional[List[Dict[str, Any]]] = None):
        self.bot_name = bot_name
        self.memory = ConversationMemory()
        self.ctx_builder = ContextBuilder(app_purpose=app_purpose, user_profile=user_profile)
        self.tools = tools or []
        self.tool_router = ToolRouter(self.tools)
        self.llm = GeminiClient()
        self.app_purpose = app_purpose

    def _system(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(bot_name=self.bot_name, app_purpose=self.app_purpose)

    def _maybe_plan_tools(self, user_message: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        # Lightweight intent/slot extraction. Replace with an LLM call if you want.
        text = user_message.lower()
        if any(k in text for k in ["hotel", "stay", "accommodation"]):
            # try to guess city and budget
            city_match = re.search(r"in\s+([a-zA-Z ]{3,})", text)
            budget_match = re.search(r"under\s*₹?\s*(\d{3,5})", text)
            args = {}
            if city_match:
                args["city"] = city_match.group(1).strip().title()
            if budget_match:
                args["budget_per_night_inr"] = int(budget_match.group(1))
            return ("get_hotels", args)
        if any(k in text for k in ["weather", "rain", "temperature"]):
            city_match = re.search(r"in\s+([a-zA-Z ]{3,})", text)
            args = {"city": city_match.group(1).strip().title()} if city_match else {}
            return ("get_weather", args)
        return None

    def respond(self, user_message: str, hints: Optional[Dict[str, Any]] = None, tool_result: Optional[str] = None) -> str:
        user_message = user_message.strip()
        self.memory.add("user", user_message)

        ctx = self.ctx_builder.build(user_message, hints=hints)
        tools_block = render_tools_block(self.tools)
        system = self._system()

        # Auto tool planning (can be bypassed by passing tool_result explicitly)
        auto_tool = self._maybe_plan_tools(user_message)
        if auto_tool and not tool_result:
            name, args = auto_tool
            tool_result = self.tool_router.call_tool(name, args)

        prompt_msgs = render_prompt(
            system=system,
            context=ctx,
            memory_summary=self.memory.summary(),
            tools_block=tools_block,
            user_message=ctx["sanitized_user_message"],
            tool_result=tool_result,
        )

        model_answer = self.llm.generate(prompt_msgs)
        safe_answer = redact_pii(model_answer)
        self.memory.add("assistant", safe_answer)
        return safe_answer


# ------------------------------
# CLI demo (optional)
# ------------------------------

if __name__ == "__main__":
    bot = WanderBot(
        bot_name="WanderBot",
        app_purpose="Help users plan trips, find hotels, food, and tips based on season and weather.",
        user_profile={"name": "Sarayu", "home": "Madurai", "preferences": ["budget", "vegetarian"]},
        tools=[
            {
                "name": "get_hotels",
                "description": "Return hotel options for a city with budget filters.",
                "schema": {
                    "city": "str",
                    "check_in": "YYYY-MM-DD",
                    "check_out": "YYYY-MM-DD",
                    "budget_per_night_inr": "int"
                }
            },
            {
                "name": "get_weather",
                "description": "Return weather for a city and date.",
                "schema": {"city": "str", "date": "YYYY-MM-DD"}
            }
        ]
    )

    print("WanderBot dynamic prompt demo. Type 'exit' to quit.\n")
    while True:
        try:
            msg = input("You: ")
        except EOFError:
            break
        if not msg or msg.lower() == "exit":
            break
        reply = bot.respond(msg)
        print("\nBot:\n" + reply + "\n")
