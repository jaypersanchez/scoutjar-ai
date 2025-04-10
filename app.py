from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
import os
from flask_cors import CORS
import requests
from dotenv import load_dotenv
load_dotenv()

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
        return jsonify({"matches": [], "suggestion": "No talents available at the moment."})

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
            tid, uid, name, email, resume, bio, exp, skills, location, availability = talents[i]

            # Strip PII for OpenAI explanation
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
                "user_id": uid,
                "name": name,
                "email": email,
                "location": location,
                "availability": availability,
                "skills": skills,
                "explanation": explanation
            })

    matches.sort(key=lambda x: -x['match_score'])

    # 🔥 NEW PART: If no matches found, ask OpenAI to suggest improvements
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
                    "temperature": 0.7
                }
            )
            suggestion_response.raise_for_status()
            suggestion = suggestion_response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print("OpenAI suggestion failed:", e)
            suggestion = "Try adjusting your search keywords for better results."

        return jsonify({"matches": [], "suggestion": suggestion})

    # Normal successful return
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

    skills_text = response.json()["choices"][0]["message"]["content"]
    return jsonify({"suggested_skills": skills_text.strip()})


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


if __name__ == '__main__':
    port = int(os.getenv("FLASK_PORT", 5000))
    #app.run(debug=True, port=port)
    app.run(host='0.0.0.0', port=5001)