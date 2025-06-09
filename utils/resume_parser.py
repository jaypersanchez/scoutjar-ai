import os
import json
import requests
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from docx import Document

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        elif ext == ".docx":
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            print(f"❌ Unsupported file type: {ext}")
            return None
    except Exception as e:
        print(f"🔥 Failed to extract text: {e}")
        return None


def parse_resume_to_fields(resume_text):
    prompt = f"""
You are an expert AI resume parser.

Analyze the resume text and semantically extract structured data, even when headings vary (e.g. "Experience", "Work History", "Professional Journey").

Return a JSON object with the following fields:

{{
  "bio": "Professional summary or career objective if present.",
  "skills": "Comma-separated list of technical and soft skills.",
  "experience": "Summarized work history with job titles, companies, and what the candidate did.",
  "education": "Highest degree or academic record.",
  "location": "City and/or country if mentioned.",
  "desired_salary": "Numeric or salary range if present.",
  "work_preferences": "Remote, On-site, or Hybrid if described.",
  "employment_type": "Full-time, Contract, Internship, etc.",
  "availability": "Immediate, 1 Month, etc.",
  "industry_experience": "Comma-separated list of industries the candidate worked in.",
  "years_experience": "Total number of years of experience, numeric."
}}

Work history should be formatted like this:
- Job Title at Company Name (Years or Dates)
  - One sentence summary of role or impact.

If a field is not found, use an empty string.

Resume:
{resume_text[:4000]}
"""




    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    })

    try:
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print("❌ Resume parsing failed:", e)
        return {}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python resume_parser.py path_to_resume.[pdf|docx|txt]")
        exit(1)

    file_path = sys.argv[1]
    resume_text = extract_text_from_file(file_path)

    if not resume_text:
        print("❌ Could not extract text.")
        exit(1)

    result = parse_resume_to_fields(resume_text)
    print(json.dumps(result, indent=2))
