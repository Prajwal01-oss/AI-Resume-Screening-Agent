import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def summarize_match(job_description: str, resume_text: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "GROQ_API_KEY not configured. "
            "Please add your API key to the environment before enabling LLM reasoning."
        )

    client = Groq(api_key=api_key)
    prompt = (
        "You are evaluating a resume against a job description. "
        "Provide a concise assessment with strengths, missing skills, and a recommendation.\n\n"
        f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
