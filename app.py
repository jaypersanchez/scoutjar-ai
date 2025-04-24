from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
import os
from flask_cors import CORS
import requests
from dotenv import load_dotenv
load_dotenv()
from utils.explanation import generate_match_explanation
import time
import re
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
from datetime import datetime, timedelta
import random

app = Flask(__name__)
CORS(app)

# Config: Set these as environment variables or change directly
DB_NAME = os.getenv("DB_NAME", "scoutjar")
DB_USER = os.getenv("DB_USER", "youruser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "yourpassword")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# This endpoint is to match a job posted in jobs table to the talent user
@app.route('/match-jobs', methods=['POST'])
def match_jobs_for_talent():
    data = request.json
    talent_id = data.get("talent_id")

    if not talent_id:
        return jsonify({"error": "Missing talent_id"}), 400

    print("🔍 Matching jobs for talent_id:", talent_id)

    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()

        # Step 1: Get the talent profile for query vector
        cursor.execute("""
            SELECT resume, bio, experience, skills
            FROM talent_profiles
            WHERE talent_id = %s;
        """, (talent_id,))
        talent = cursor.fetchone()

        if not talent:
            return jsonify({"error": "Talent not found"}), 404

        resume, bio, experience, skills = talent
        talent_query = f"{resume or ''} {bio or ''} {experience or ''} {' '.join(skills or [])}"

        # Step 2: Get all jobs not applied for
        cursor.execute("""
            SELECT j.job_id, j.job_title, j.job_description, j.required_skills, j.recruiter_id
            FROM jobs j
            WHERE j.job_id NOT IN (
                SELECT job_id FROM job_applications WHERE talent_id = %s
            );
        """, (talent_id,))
        jobs = cursor.fetchall()
        cursor.close()
        conn.close()

        if not jobs:
            print("⚠️ No available jobs for this talent.")
            return jsonify({"matches": []})

        job_docs = [
            f"{title} {desc} {' '.join(skills or [])}" for (_, title, desc, skills, _) in jobs
        ]

        tfidf = TfidfVectorizer(stop_words='english')
        vectors = tfidf.fit_transform([talent_query] + job_docs)
        scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        matches = []
        for i, score in enumerate(scores):
            if score >= 0.1:
                job = jobs[i]
                matches.append({
                    "job_id": job[0],
                    "title": job[1],
                    "description": job[2],
                    "skills_required": job[3],
                    "recruiter_id": job[4],
                    "match_score": round(score * 100, 2)
                })

        matches.sort(key=lambda x: -x['match_score'])
        print(f"✅ Matched {len(matches)} jobs for talent_id {talent_id}")
        return jsonify({"matches": matches})

    except Exception as e:
        print("🔥 Error in /match-jobs:", e)
        return jsonify({"error": "Failed to match jobs"}), 500




# Return a list of jobs the talent has applied for
@app.route('/applied-jobs', methods=['POST'])
def get_applied_jobs():
    data = request.json
    talent_id = data.get("talent_id")

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    cursor.execute("""
        SELECT j.job_id, j.job_title, j.job_description, j.required_skills, j.recruiter_id,
            (SELECT COUNT(*) FROM job_applications WHERE job_id = j.job_id) as applicant_count
        FROM jobs j
        JOIN job_applications ja ON j.job_id = ja.job_id
        WHERE ja.talent_id = %s
    """, (talent_id,))
    
    jobs = cursor.fetchall()
    cursor.close()
    conn.close()

    applied_jobs = []
    for row in jobs:
        job_id, title, description, required_skills, recruiter_id, applicant_count = row
        applied_jobs.append({
            "job_id": job_id,
            "job_title": title,
            "job_description": description,
            "required_skills": required_skills or [],
            "recruiter_id": recruiter_id,
            "applicant_count": applicant_count
        })
    
    return jsonify({"applied_jobs": applied_jobs})

# This endpoint is used to explain to the scout talent how a talent matches with their job post
@app.route("/explain-match", methods=["POST"])
def explain_match():
    data = request.json
    job = data.get("job", {})
    talent = data.get("talent", {})

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

    response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)

    if response.status_code != 200:
        return jsonify({"error": "Failed to get response from OpenAI", "details": response.text}), 500

    explanation = response.json()["choices"][0]["message"]["content"]
    return jsonify({ "explanation": explanation })
    
# This endpoint is used for semantic style search 
'''@app.route('/search-talents', methods=['POST'])
def search_talents():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    print("🔍 Scout Search Query:", query)

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # ✅ Include tp.user_id in the SELECT
    cursor.execute("""
        SELECT tp.talent_id, tp.user_id, up.full_name, up.email, tp.resume, tp.bio, tp.experience,
               tp.skills, tp.location, tp.availability
        FROM public.talent_profiles tp
        JOIN public.user_profiles up ON tp.user_id = up.user_id;
    """)
    talents = cursor.fetchall()
    cursor.close()
    conn.close()

    if not talents:
        return jsonify({"matches": []})

    # Build document corpus: query + all talent docs
    docs = [query]
    for talent in talents:
        resume, bio, exp = talent[4], talent[5], talent[6]
        combined_text = f"{resume or ''} {bio or ''} {exp or ''}"
        docs.append(combined_text)

    # Compute similarity
    tfidf = TfidfVectorizer(stop_words='english')
    vectors = tfidf.fit_transform(docs)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    threshold = 0.1
    matches = []

    for i, score in enumerate(scores):
        if score >= threshold:
            # ✅ Unpack user_id in the right order
            tid, uid, name, email, resume, bio, exp, skills, location, availability = talents[i]

            # Strip PII for OpenAI
            stripped_info = {
                "resume": resume,
                "bio": bio,
                "experience": exp,
                "skills": skills,
                "availability": availability
            }

            prompt = f"""
You are an AI talent scout. A recruiter is looking for this: "{query}"

Here is a candidate's anonymized profile:
Resume: {stripped_info['resume']}
Bio: {stripped_info['bio']}
Experience: {stripped_info['experience']}
Skills: {', '.join(stripped_info['skills'])}
Availability: {stripped_info['availability']}

Explain in 3–4 sentences why this candidate might be a good match.
"""

            explanation = ""
            try:
                openai_response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "system", "content": "You are a helpful recruiter assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7
                    }
                )
                openai_response.raise_for_status()
                explanation = openai_response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print("OpenAI explanation failed:", e)
                explanation = "Explanation not available."

            matches.append({
                "match_score": round(score * 100, 2),
                "talent_id": tid,
                "user_id": uid,  # ✅ Now included in the result
                "name": name,
                "email": email,
                "location": location,
                "availability": availability,
                "skills": skills,
                "explanation": explanation
            })

    matches.sort(key=lambda x: -x['match_score'])
    return jsonify({"matches": matches})
'''

# This endpoint is used to return a match of talents based on scout talent job post details
@app.route('/jobs', methods=['POST'])
def match_talents():
    job = request.json
    job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('skills', '')}"
    print("🔥 Incoming job:", job)
    print("🧠 Combined job_text for TF-IDF:", job_text)

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    cursor.execute("""
        SELECT tp.talent_id, up.full_name, tp.resume, tp.bio, tp.experience, tp.skills, tp.location, tp.availability
        FROM public.talent_profiles tp
        JOIN public.user_profiles up ON tp.user_id = up.user_id;
    """)
    talents = cursor.fetchall()
    cursor.close()
    conn.close()

    if not talents:
        return jsonify({"matches": []})

    docs = [job_text]
    for talent in talents:
        resume, bio, exp = talent[2], talent[3], talent[4]
        combined_text = f"{resume or ''} {bio or ''} {exp or ''}"
        docs.append(combined_text)

    tfidf = TfidfVectorizer(stop_words='english')
    vectors = tfidf.fit_transform(docs)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    threshold = 0.1
    matches = []

    for i, score in enumerate(scores):
        if score >= threshold:
            tid, name, resume, bio, exp, skills, location, availability = talents[i]
            matches.append({
                "match_score": round(score * 100, 2),
                "talent_id": tid,
                "name": name,
                "resume": resume,
                "bio": bio,
                "experience": exp,
                "skills": skills,
                "location": location,
                "availability": availability
            })

    matches.sort(key=lambda x: -x['match_score'])
    return jsonify({"matches": matches})
    
# search talent with suggestion 
@app.route('/search-talents', methods=['POST'])
def search_talents():
    import json  # Make sure you have this at the top

    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    print("🔍 Scout Search Query:", query)

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    cursor.execute("""
        SELECT tp.talent_id, tp.user_id, up.full_name, up.email, tp.resume, tp.bio, tp.experience,
               tp.skills, tp.location, tp.availability
        FROM public.talent_profiles tp
        JOIN public.user_profiles up ON tp.user_id = up.user_id;
    """)
    talents = cursor.fetchall()
    cursor.close()
    conn.close()

    if not talents:
        return jsonify({"matches": [], "suggestion": json.dumps({
            "advice": "No talents available at the moment.",
            "refined_prompt": ""
        })})

    docs = [query]
    for talent in talents:
        resume, bio, exp = talent[4], talent[5], talent[6]
        combined_text = f"{resume or ''} {bio or ''} {exp or ''}"
        docs.append(combined_text)

    tfidf = TfidfVectorizer(stop_words='english')
    vectors = tfidf.fit_transform(docs)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    threshold = 0.1
    matches = []

    for i, score in enumerate(scores):
        if score >= threshold:
            tid, uid, name, email, resume, bio, exp, skills, location, availability = talents[i]
            matches.append({
                "match_score": round(score * 100, 2),
                "talent_id": tid,
                "user_id": uid,
                "name": name,
                "email": email,
                "location": location,
                "availability": availability,
                "skills": skills,
                "explanation": ""
            })

    matches.sort(key=lambda x: -x['match_score'])

    # 🔥 If no matches, generate suggestion
    if not matches:
        try:
            suggest_prompt = f"""
You are Lookk, a helpful AI scout assistant.

A scout entered the following talent search: "{query}"

No exact matches were found.

Please do the following:
1. Give a polite and brief advice (1-2 sentences) on how to broaden or improve the search.
2. Create a new refined version of the scout's original search query, making it more likely to find matching talent.
- Keep it realistic, generalize if needed (e.g., "Product Manager with technical background")
- Keep it under 30 words if possible.

Return it in this exact JSON format:
{{
  "advice": "...your advice here...",
  "refined_prompt": "...your better search prompt here..."
}}

Be concise and friendly.
"""
            suggestion_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": suggest_prompt}
                    ],
                    "temperature": 0.4
                }
            )
            suggestion_response.raise_for_status()
            raw_suggestion = suggestion_response.json()["choices"][0]["message"]["content"]

            try:
                # Attempt to parse OpenAI response as JSON
                parsed_suggestion = json.loads(raw_suggestion)
            except json.JSONDecodeError:
                # If OpenAI returns plain text, wrap it manually
                parsed_suggestion = {
                    "advice": raw_suggestion.strip(),
                    "refined_prompt": ""
                }

            suggestion = json.dumps(parsed_suggestion)

        except Exception as e:
            print("🔥 OpenAI Suggestion Error:", e)
            suggestion = json.dumps({
                "advice": "Try adjusting your search keywords for better results.",
                "refined_prompt": ""
            })

        return jsonify({"matches": [], "suggestion": suggestion})

    # 🔥 Normal return if there are matches
    return jsonify({"matches": matches})

@app.route('/recruiter-info/<int:job_id>', methods=['GET'])
def get_recruiter_info(job_id):
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT up.full_name, tr.company_name, tr.company_logo
            FROM public.jobs j
            JOIN public.talent_recruiters tr ON j.recruiter_id = tr.recruiter_id
            JOIN public.user_profiles up ON tr.user_id = up.user_id
            WHERE j.job_id = %s;
        """, (job_id,))

        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            return jsonify({"error": "Recruiter not found for this job"}), 404

        full_name, company_name, company_logo = row

        return jsonify({
            "full_name": full_name,
            "company": company_name,
            "profile_image": company_logo
        })

    except Exception as e:
        print("🔥 Error in /recruiter-info:", e)
        return jsonify({"error": "Failed to retrieve recruiter info"}), 500


@app.route('/suggest-skills', methods=['POST'])
def suggest_skills():
    data = request.json
    job_title = data.get("job_title", "")
    job_description = data.get("job_description", "")

    if not job_title and not job_description:
        return jsonify({"error": "Job title or description required."}), 400

    prompt = f"""
You are an expert recruiter assistant.

Based on the given Job Title and Job Description, infer and generate a realistic list of required job skills even if the description is short or vague.

Always provide a comma-separated list of 5 to 10 relevant skills typically needed for the role.

Only return the skills list, without any explanation or extra text.

Job Title: {job_title}
Job Description: {job_description}
"""

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)

    if response.status_code != 200:
        return jsonify({"error": "Failed to get skill suggestion from OpenAI", "details": response.text}), 500

    #skills_text = response.json()["choices"][0]["message"]["content"]
    #return jsonify({"suggested_skills": skills_text.strip()})
    raw_text = response.json()["choices"][0]["message"]["content"].strip()

    # Split on newline or comma
    if "\n" in raw_text:
        lines = re.split(r"[-•\d.]*\s*", raw_text)
        skills = [s.strip() for s in lines if s.strip()]
    else:
        skills = [s.strip() for s in raw_text.split(",") if s.strip()]

    normalized = ", ".join(skills)
    return jsonify({"suggested_skills": normalized})


@app.route('/ai-match-talents', methods=['POST'])
def ai_match_talents():
    data = request.json

    job_title = data.get("job_title", "")
    job_description = data.get("job_description", "")
    required_skills = data.get("required_skill", "")
    industry_experience = data.get("industry_experience", "")
    years_experience = data.get("years_experience", 0)
    match_threshold = data.get("match_percentage", 50) / 100.0

    job_vector = f"{job_title} {job_description} {required_skills} {industry_experience} {years_experience}"

    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()

        cursor.execute("""
            SELECT tp.talent_id, tp.resume, tp.bio, tp.experience, tp.skills, tp.industry_experience,
                   tp.years_experience, tp.desired_salary, tp.location, tp.work_preferences,
                   tp.availability, up.full_name, up.email
            FROM public.talent_profiles tp
            JOIN public.user_profiles up ON tp.user_id = up.user_id;
        """)

        talents = cursor.fetchall()
        cursor.close()
        conn.close()

    except Exception as e:
        print("🔥 Error fetching talent profiles:", e)
        return jsonify({"error": "Database error"}), 500

    job_doc = [job_vector]
    talent_docs = [
        f"{resume or ''} {bio or ''} {exp or ''} {' '.join(skills or [])} {' '.join(industry or [])} {years or 0}"
        for (_, resume, bio, exp, skills, industry, years, _, _, _, _, _, _) in talents
    ]

    tfidf = TfidfVectorizer(stop_words='english')
    vectors = tfidf.fit_transform(job_doc + talent_docs)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    top_matches = sorted(
        [(i, score) for i, score in enumerate(scores) if score >= match_threshold],
        key=lambda x: -x[1]
    )[:25]  # Top 25 for now

    results = []
    for i, score in top_matches:
        tid, resume, bio, exp, skills, industry, years, salary, location, work_preferences, availability, name, email = talents[i]

        try:
            start_time = time.time()
            explanation = generate_match_explanation(
                {
                    "title": job_title,
                    "description": job_description,
                    "skills": required_skills
                },
                {
                    "name": name,
                    "resume": resume,
                    "bio": bio,
                    "experience": exp,
                    "skills": skills or []
                }
            )
            elapsed = round(time.time() - start_time, 2)
            print(f"⏱ Explanation for talent_id {tid} took {elapsed} seconds")
        except Exception as e:
            print(f"❌ Failed to get explanation for talent_id {tid}:", e)
            explanation = "Explanation not available."

        results.append({
            "talent_id": tid,
            "full_name": name,
            "email": email,
            "resume": resume,
            "bio": bio,
            "experience": exp,
            "skills": skills,
            "industry_experience": industry,
            "years_experience": years,
            "desired_salary": salary,
            "location": location,
            "work_preferences": work_preferences,
            "availability": availability,
            "match_score": round(score * 100, 2),
            "explanation": explanation
        })

    results.sort(key=lambda x: -x["match_score"])
    return jsonify({"matches": results})

@app.route("/job-titles/all", methods=["GET"])
def get_all_job_titles():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT job_title FROM job_titles ORDER BY job_title ASC;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        return jsonify([row[0] for row in rows])
    except Exception as e:
        print("🔥 Error fetching job titles:", e)
        return jsonify([]), 500

# This is for short list of talents that recruiter have approach but not necessarily tied to a job
@app.route("/ai-shortlist", methods=["POST"])
def ai_shortlist():
    data = request.json
    recruiter_id = data.get("recruiter_id")
    talent_id = data.get("talent_id")
    job_id = data.get("job_id")  # optional

    if not recruiter_id or not talent_id:
        return jsonify({"error": "Missing recruiter_id or talent_id"}), 400

    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT
        )
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO ai_shortlisted (recruiter_id, talent_id, job_id)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, (recruiter_id, talent_id, job_id))

        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Talent AI-shortlisted successfully."})

    except Exception as e:
        print("🔥 AI Shortlist error:", e)
        return jsonify({"error": "Shortlist failed"}), 500

@app.route("/ai-shortlisted-candidates", methods=["POST"])
def get_ai_shortlisted_candidates():
    data = request.json
    recruiter_id = data.get("recruiter_id")

    if not recruiter_id:
        return jsonify({"error": "Missing recruiter_id"}), 400

    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.id AS shortlist_id, a.talent_id, a.created_at AS added_at,
                   up.full_name, up.email
            FROM ai_shortlisted a
            JOIN talent_profiles tp ON a.talent_id = tp.talent_id
            JOIN user_profiles up ON tp.user_id = up.user_id
            WHERE a.recruiter_id = %s;
        """, (recruiter_id,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        return jsonify([
            {
                "shortlist_id": row[0],
                "talent_id": row[1],
                "added_at": row[2],
                "full_name": row[3],
                "email": row[4]
            } for row in rows
        ])
    except Exception as e:
        print("🔥 AI Shortlist Fetch Error:", e)
        return jsonify({"error": "Failed to fetch AI-shortlisted candidates"}), 500

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Fictional resume creator for data testing
def extract_text_from_file(file_path, extension):
    try:
        if extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif extension == 'pdf':
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() or '' for page in reader.pages])
        elif extension == 'docx':
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            return None
    except Exception as e:
        print(f"🔥 Failed to extract text: {e}")
        return None

@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    talent_id = request.form.get("talent_id")
    file = request.files.get("file")

    if not talent_id or not file:
        return jsonify({"error": "Missing file or talent_id"}), 400

    filename = secure_filename(file.filename)
    extension = filename.rsplit('.', 1)[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type"}), 400

    temp_path = os.path.join("/tmp", filename)
    file.save(temp_path)

    extracted_text = extract_text_from_file(temp_path, extension)
    os.remove(temp_path)

    if not extracted_text:
        return jsonify({"error": "Failed to extract text"}), 500

    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE talent_profiles
            SET resume = %s
            WHERE talent_id = %s
        """, (extracted_text.strip(), talent_id))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"message": "Resume updated successfully"})
    except Exception as e:
        print("🔥 DB Error:", e)
        return jsonify({"error": "Database update failed"}), 500

@app.route("/generate-fictional-resumes", methods=["POST"])
def generate_fictional_resumes():
    try:
        talent_id = request.json.get("talent_id") if request.is_json else None

        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT
        )
        cursor = conn.cursor()

        if talent_id:
            cursor.execute("""
                SELECT tp.talent_id, up.full_name, up.email, tp.bio, tp.experience, tp.skills, tp.industry_experience,
                       tp.education, tp.location, tp.available_from
                FROM talent_profiles tp
                JOIN user_profiles up ON tp.user_id = up.user_id
                WHERE tp.talent_id = %s;
            """, (talent_id,))
        else:
            cursor.execute("""
                SELECT tp.talent_id, up.full_name, up.email, tp.bio, tp.experience, tp.skills, tp.industry_experience,
                       tp.education, tp.location, tp.available_from
                FROM talent_profiles tp
                JOIN user_profiles up ON tp.user_id = up.user_id;
            """)

        records = cursor.fetchall()

        companies = ["NeoTech Solutions", "Bluewave Systems", "CodeCraft Inc.", "NextGen Innovations", "SkyRocket Labs"]
        roles = ["Software Engineer", "Backend Developer", "Frontend Developer", "Fullstack Engineer", "DevOps Specialist"]
        actions = [
            "Led the development of key platform features",
            "Collaborated with cross-functional teams to deliver scalable solutions",
            "Optimized performance and reduced load times by 40%",
            "Integrated third-party APIs and services",
            "Implemented CI/CD pipelines to streamline deployment",
            "Mentored junior developers and conducted code reviews"
        ]

        for (tid, name, email, bio, exp, skills, industries, edu, loc, avail) in records:
            num_jobs = random.randint(3, 5)
            start_year = datetime.now().year - num_jobs
            work_sections = []

            for i in range(num_jobs):
                job_title = random.choice(roles)
                company = random.choice(companies)
                city = loc or "Remote"
                start = datetime(start_year + i, random.randint(1, 6), 1)
                end = start + timedelta(days=365 * random.randint(1, 2))
                responsibilities = random.sample(actions, 2)
                bullet_points = "\n   - " + "\n   - ".join(responsibilities)
                work_sections.append(f"{job_title} at {company}\n{start.strftime('%b %Y')} – {end.strftime('%b %Y')} | {city}{bullet_points}")

            summary = bio.strip() if bio else "Motivated and skilled professional with experience in diverse technology domains."
            skill_str = ", ".join(skills or ["Communication", "Teamwork", "Problem-solving"])
            industry_str = ", ".join(industries or ["Technology"])
            education_str = edu or "Bachelor's Degree"
            location_str = loc or "Remote"
            available_str = avail.strftime("%B %d, %Y") if avail else "N/A"
            exp_context = exp.strip() if exp else f"Contributed significantly in the {industry_str} domain."

            resume = f"""
{name}
{location_str} | {email}

Summary:
{summary}

Core Competencies:
{skill_str}

Work Experience:
{chr(10).join(work_sections)}

Additional Experience:
{exp_context}

Education:
{education_str}

Availability:
{available_str}
""".strip()

            cursor.execute("""
                UPDATE talent_profiles SET resume = %s WHERE talent_id = %s;
            """, (resume, tid))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"message": "Enhanced fictional resumes generated successfully."})

    except Exception as e:
        print("🔥 Error generating enhanced resumes:", e)
        return jsonify({"error": "Internal server error."}), 500

if __name__ == '__main__':
    port = int(os.getenv("FLASK_PORT", 5001))
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    app.run(host=host, port=port)
    
