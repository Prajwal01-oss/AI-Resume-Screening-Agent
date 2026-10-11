import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.embedding import build_embedding
from src.llm import generate_cover_letter, improve_bullets, review_ats, summarize_match
from src.parser import parse_resume
from src.similarity import compute_similarity
from src.utils import ensure_directory

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(title="Resume Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def analyze_resumes(job_description: str, resume_dir: str, output_dir: str) -> list[dict]:
    ensure_directory(output_dir)
    jd_text = job_description.strip()
    jd_embedding = build_embedding(jd_text)["embedding"]

    results = []
    for resume_file in sorted(Path(resume_dir).glob("*")):
        if not resume_file.is_file():
            continue
        resume_text = parse_resume(str(resume_file))
        resume_embedding = build_embedding(resume_text)["embedding"]
        similarity = compute_similarity(jd_embedding, resume_embedding)
        score = round(similarity * 100, 2)
        recommendation = "Recommended for Interview" if score >= 75 else "Needs Review"
        reasoning = summarize_match(jd_text, resume_text)

        result = {
            "candidate": resume_file.stem,
            "score": score,
            "recommendation": recommendation,
            "reason": reasoning,
        }
        results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)

    df = pd.DataFrame(results)
    out_dir = Path(output_dir)
    df.to_csv(out_dir / "results.csv", index=False)
    with open(out_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Wrote {len(results)} results to {out_dir}")
    return results


def analyze_uploaded_resumes(job_description: str, uploaded_files: list[UploadFile]) -> list[dict]:
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        saved_files = []
        for uploaded_file in uploaded_files:
            if not uploaded_file.filename:
                continue
            target_path = temp_dir / uploaded_file.filename
            target_path.write_bytes(uploaded_file.file.read())
            saved_files.append(target_path)

        if not saved_files:
            raise HTTPException(status_code=400, detail="Please upload at least one valid resume file.")

        jd_embedding = build_embedding(job_description)["embedding"]
        results = []
        for resume_path in sorted(saved_files):
            resume_text = parse_resume(str(resume_path))
            resume_embedding = build_embedding(resume_text)["embedding"]
            similarity = compute_similarity(jd_embedding, resume_embedding)
            score = round(similarity * 100, 2)
            recommendation = "Recommended for Interview" if score >= 75 else "Needs Review"
            reasoning = summarize_match(job_description, resume_text)
            results.append(
                {
                    "candidate": resume_path.stem,
                    "score": score,
                    "recommendation": recommendation,
                    "reason": reasoning,
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results


TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"


class ImproveBulletsRequest(BaseModel):
    bullets: str


class CoverLetterRequest(BaseModel):
    company: str
    role: str
    manager: str = ""
    summary: str = ""
    bullets: str = ""
    skills: str = ""


class ReviewAtsRequest(BaseModel):
    job_description: str
    resume_text: str


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    if TEMPLATE_PATH.exists():
        content = TEMPLATE_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>Resume Studio</h1><p>Template file not found.</p>", status_code=500)


@app.post("/api/ai/improve-bullets")
async def improve_bullets_endpoint(payload: ImproveBulletsRequest):
    if not payload.bullets.strip():
        raise HTTPException(status_code=400, detail="Please provide bullets to improve.")
    try:
        improved = improve_bullets(payload.bullets)
        return JSONResponse({"improved": improved})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ai/cover-letter")
async def cover_letter_endpoint(payload: CoverLetterRequest):
    if not payload.company.strip() or not payload.role.strip():
        raise HTTPException(status_code=400, detail="Company and role are required.")
    try:
        letter = generate_cover_letter(
            company=payload.company,
            role=payload.role,
            manager=payload.manager,
            resume_summary=payload.summary,
            bullets=payload.bullets,
            skills=payload.skills,
        )
        return JSONResponse({"cover_letter": letter})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ai/review-ats")
async def review_ats_endpoint(payload: ReviewAtsRequest):
    if not payload.job_description.strip() or not payload.resume_text.strip():
        raise HTTPException(status_code=400, detail="Job description and resume text are required.")
    try:
        review = review_ats(payload.job_description, payload.resume_text)
        return JSONResponse({"review": review})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/analyze")
async def analyze_endpoint(
    job_description: str = Form(...),
    files: list[UploadFile] = File(...),
):
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Please enter a job description.")
    results = analyze_uploaded_resumes(job_description, files)
    return JSONResponse({"results": results})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    job_description = "Software Engineer with Python, AWS, Docker, and Git"
    resume_dir = os.path.join("data", "resumes")
    output_dir = os.path.join("data", "output")
    results = analyze_resumes(job_description, resume_dir, output_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
