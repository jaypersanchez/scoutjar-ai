# ✅ STEP 1: Create this file as `utils/explanation.py`

import os
import requests

def generate_match_explanation(job: dict, talent: dict) -> str:
    prompt = f"""
You are an expert recruiter assistant. You are helping evaluate a candidate for a job.

Job Title: {job.get("title")}
Job Description: {job.get("description")}
Required Skills: {job.get("skills")}

Candidate Name: {talent.get("name")}
Resume: {talent.get("resume")}
Bio: {talent.get("bio")}
Experience: {talent.get("experience")}
Skills: {', '.join(talent.get('skills', []))}

Based on this information, explain in 2-3 sentences why this candidate is a good match for the job. Focus on alignment of experience and skills.
"""

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        explanation = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ OpenAI error in generate_match_explanation:", e)
        explanation = "Explanation not available."

    return explanation
