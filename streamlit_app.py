import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.embedding import build_embedding
from src.llm import summarize_match
from src.parser import parse_resume
from src.similarity import compute_similarity

st.set_page_config(page_title="AI Resume Screening Agent", layout="wide")
st.title("AI Resume Screening Agent")
st.write("Upload a job description and one or more resumes to generate a ranked shortlist.")

job_description = st.text_area(
    "Job Description",
    value="Software Engineer with Python, AWS, Docker, and Git",
    height=150,
)
resume_files = st.file_uploader(
    "Upload resumes",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

if st.button("Analyze Resumes"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    elif not resume_files:
        st.warning("Please upload at least one resume.")
    else:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            saved_files = []
            for uploaded_file in resume_files:
                target_path = temp_dir / uploaded_file.name
                target_path.write_bytes(uploaded_file.getvalue())
                saved_files.append(target_path)

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
            if results:
                df = pd.DataFrame(results)
                st.success("Analysis complete.")
                st.dataframe(df, use_container_width=True)

                csv_data = df.to_csv(index=False).encode("utf-8")
                json_data = json.dumps(results, indent=2).encode("utf-8")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download CSV",
                        data=csv_data,
                        file_name="results.csv",
                        mime="text/csv",
                    )
                with col2:
                    st.download_button(
                        "Download JSON",
                        data=json_data,
                        file_name="results.json",
                        mime="application/json",
                    )
            else:
                st.info("No resume files were processed.")
