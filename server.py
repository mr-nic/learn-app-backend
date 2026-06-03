from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import re
import os
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

GENERATE_PROMPT = lambda transcript: f"""You are a learning content generator. Given this transcript, produce a structured learning package.

TRANSCRIPT:
---
{transcript}
---

Respond with ONLY a valid JSON object, no markdown, no backticks, no preamble. Structure:

{{
  "title": "Short compelling course title",
  "source_hint": "Brief description of what this content is about (1 sentence)",
  "summary": {{
    "points": ["key concept 1", "key concept 2", "key concept 3", "key concept 4", "key concept 5", "key concept 6"]
  }},
  "flashcards": [
    {{ "front": "Term or concept", "back": "Clear definition or explanation" }}
  ],
  "quiz": [
    {{
      "q": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct": 0,
      "explanation": "Why the correct answer is right"
    }}
  ]
}}

Rules:
- summary: exactly 6 points, each under 15 words
- flashcards: exactly 8 cards
- quiz: exactly 5 questions, each with exactly 4 options, correct is 0-indexed
- All content must come from the transcript only
- JSON must be valid and parseable"""


class TranscriptRequest(BaseModel):
    url: str


class GenerateRequest(BaseModel):
    transcript: str


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"embed\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


@app.post("/transcript")
async def get_transcript(req: TranscriptRequest):
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        snippets = list(transcript)
        full_text = " ".join(s.text for s in snippets)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch transcript: {str(e)}")

    if len(full_text) > 8000:
        full_text = full_text[:8000] + "... [transcript trimmed]"

    return {"video_id": video_id, "transcript": full_text, "length": len(full_text)}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set on server.")

    transcript = req.transcript[:12000]

    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4000,
                "messages": [{"role": "user", "content": GENERATE_PROMPT(transcript)}],
            },
        )

    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Anthropic API error: {res.text}")

    data = res.json()
    raw = next((b["text"] for b in data.get("content", []) if b["type"] == "text"), "")
    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        import json
        parsed = json.loads(clean)
    except Exception:
        raise HTTPException(status_code=500, detail=f"JSON parse failed: {clean[:200]}")

    return parsed


@app.post("/chat")
async def chat(req: dict):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set on server.")

    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "system": req.get("system", ""),
                "messages": req.get("messages", []),
            },
        )

    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Anthropic API error: {res.text}")

    data = res.json()
    text = next((b["text"] for b in data.get("content", []) if b["type"] == "text"), "No response.")
    return {"text": text}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
