import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.embedding import build_embedding
from src.llm import summarize_match
from src.parser import parse_resume
from src.similarity import compute_similarity
from src.utils import ensure_directory

load_dotenv(Path(__file__).resolve().parent / ".env")


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


def main() -> None:
    job_description = "Software Engineer with Python, AWS, Docker, and Git"
    resume_dir = os.path.join("data", "resumes")
    output_dir = os.path.join("data", "output")
    results = analyze_resumes(job_description, resume_dir, output_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
