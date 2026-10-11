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


def improve_bullets(bullets_text: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return bullets_text

    client = Groq(api_key=api_key)
    prompt = (
        "You are an executive resume coach and ATS optimization specialist. "
        "Take the following resume bullet points and strengthen each one. "
        "Use high-impact action verbs (e.g., Architected, Accelerated, Spearheaded, Engineered), "
        "quantify results where appropriate, and keep each bullet clear, punchy, and professional. "
        "Return ONLY the rewritten bullet points, one per line, with no extra conversational commentary.\n\n"
        f"Input bullets:\n{bullets_text}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def generate_cover_letter(company: str, role: str, manager: str, resume_summary: str, bullets: str, skills: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""

    client = Groq(api_key=api_key)
    prompt = (
        "Write a compelling, professional cover letter tailored for:\n"
        f"Company: {company}\n"
        f"Role: {role}\n"
        f"Hiring Manager: {manager or 'Hiring Team'}\n\n"
        f"Candidate Summary: {resume_summary}\n"
        f"Candidate Key Achievements:\n{bullets}\n"
        f"Candidate Core Skills: {skills}\n\n"
        "Instructions: Make it persuasive, authentic, and concise (under 300 words). "
        "Do not use generic clichés. Return only the text of the letter."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def review_ats(job_description: str, resume_text: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""

    client = Groq(api_key=api_key)
    prompt = (
        "You are an expert talent acquisition recruiter and ATS scanner. "
        "Evaluate this candidate's resume against the target job description. "
        "Provide: 1) Executive Match Summary, 2) Key Strengths, 3) Critical Missing Skills/Gaps, 4) Interview Recommendation.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Resume:\n{resume_text}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()

