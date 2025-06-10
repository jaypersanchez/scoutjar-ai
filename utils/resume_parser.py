import os
import json
import requests
import sys
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from docx import Document
from datetime import datetime

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def log_debug(message):
    with open("/tmp/parser.log", "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")

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
  "education": "Extract all educational background information, including degrees, school names, certifications, online courses, and any section titled Education or mentioning Advanced Education, even if informal. Look at the bottom of the resume for any education-related content, including nontraditional education like online courses or informal studies.
",
  "location": "City and/or country if mentioned.",
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
{resume_text[:10000]}
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
        #print("✅ Raw OpenAI response (first 300 chars):", content[:300], file=sys.stderr)
        log_debug("✅ Full OpenAI response:\n" + content)
        #return json.loads(content)
        parsed = json.loads(content)
        log_debug("🎯 Parsed resume fields: " + json.dumps(parsed)[:500])
        return parsed
    except Exception as e:
        log_debug(f"❌ Resume parsing failed: {str(e)}")
        log_debug(f"❌ Raw response: {response.text}")
        print("❌ Resume parsing failed:", e, file=sys.stderr)
        print("❌ Full response text:", response.text, file=sys.stderr)
        return {}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python resume_parser.py path_to_resume.[pdf|docx|txt]")
        exit(1)

    file_path = sys.argv[1]
    resume_text = extract_text_from_file(file_path)
    print("✅ Extracted resume text length:", len(resume_text), file=sys.stderr)
    
    if not resume_text:
        print("❌ Could not extract text.")
        exit(1)

    result = parse_resume_to_fields(resume_text)
    print(json.dumps(result, indent=2))
