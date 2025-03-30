from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
import os
from flask_cors import CORS
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

if __name__ == '__main__':
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(debug=True, port=port)