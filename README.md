# AI Resume Screening Agent

## Project Overview

AI Resume Screening Agent is a Python-based prototype that reads resumes, compares them against a job description using embeddings and cosine similarity, and produces ranked output in CSV and JSON format. It can also use Groq for LLM-based reasoning when a valid API key is provided.

## Features

- Reads resume files in PDF, DOCX, TXT, and MD formats
- Extracts text from resumes and job descriptions
- Generates embeddings with Sentence Transformers
- Computes cosine similarity between the job description and each resume
- Produces ranked results with scores and recommendations
- Exports results to CSV and JSON
- Supports optional Groq-based reasoning for strengths, gaps, and recommendation

## Architecture

1. Job description input
2. Resume file ingestion
3. Text extraction
4. Embedding generation
5. Similarity scoring
6. Optional LLM reasoning
7. CSV/JSON export and ranking display

## Installation

```bash
python -m pip install -r requirements.txt
```

## Setup

1. Create a virtual environment
2. Install dependencies
3. Add your API key to `.env`
4. Place resume files in `data/resumes/`
5. Run the app with `python app.py`

## Preview / Demo

The web app is currently available at:

- Local preview: http://localhost:8501
- Network preview: http://192.168.29.105:8501

Open either link in your browser to try the project locally.

## API Key Configuration

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_api_key_here
```

If no key is provided, the app will still run and generate similarity-based output, but LLM reasoning will be skipped with a fallback message.

## Run Instructions

```bash
python app.py
```

This generates output in:

- `data/output/results.csv`
- `data/output/results.json`

## Sample Inputs

Example job description:

```text
Software Engineer with Python, AWS, Docker, and Git
```

Example resume file:

```text
Experienced Python developer with AWS, Docker, Git, and CI/CD experience.
```

## Sample Outputs

Example JSON output:

```json
[
  {
    "candidate": "sample_resume",
    "score": 80.67,
    "recommendation": "Recommended for Interview",
    "reason": "GROQ_API_KEY not configured. Please add your API key to the environment before enabling LLM reasoning."
  }
]
```

## Design Decisions

- Sentence Transformers was chosen for fast semantic similarity scoring.
- Cosine similarity was used because it works well for comparing embedding vectors.
- The app keeps the workflow modular with separate modules for parsing, embedding, similarity, LLM reasoning, and export.

## Limitations

- The current version is a prototype and not a production-grade hiring platform.
- LLM-based reasoning requires a valid Groq API key.
- Resume parsing quality depends on the file format and document quality.

## Future Improvements

- Add a Streamlit UI for uploading JD and resumes
- Support batch processing and ranking dashboards
- Improve parsing for complex PDF layouts
- Add explainable scoring and recruiter-friendly summaries
- Integrate with a database or cloud storage
