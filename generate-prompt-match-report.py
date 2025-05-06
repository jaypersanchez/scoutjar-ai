import os
import psycopg2
import requests
import json
import csv
from dotenv import load_dotenv

load_dotenv()

# DB connection
DB_NAME = os.getenv("DB_NAME", "scoutjar")
DB_USER = os.getenv("DB_USER", "youruser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "yourpassword")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

SEARCH_ENDPOINT = "http://localhost:5001/search-talents"

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cursor = conn.cursor()

cursor.execute("""
    SELECT talent_id, resume, bio, experience, skills
    FROM talent_profiles;
""")
talents = cursor.fetchall()
conn.close()

output_file = "talent_prompt_matching_report.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Talent ID", "Prompt", "Top Match Talent ID", "Match Score", "Matched Self?", "Explanation"])

    success_count = 0

    for tid, resume, bio, experience, skills in talents:
        resume = resume or ""
        bio = bio or ""
        experience = experience or ""
        skills_str = ", ".join(skills or [])

        prompt = f"""
Looking for a professional with the following background:
{bio.strip()}

Should have experience in:
{experience.strip()}

Skills must include:
{skills_str}
"""

        try:
            resp = requests.post(SEARCH_ENDPOINT, json={"query": prompt.strip()})
            data = resp.json()
            matches = data.get("matches", [])

            if not matches:
                writer.writerow([tid, prompt.strip(), "None", 0, "No", "No matches returned"])
                continue

            top_match = matches[0]
            match_score = top_match.get("match_score", 0)
            matched_id = top_match.get("talent_id")
            matched_self = "Yes" if matched_id == tid and match_score >= 80 else "No"
            explanation = top_match.get("explanation", "")

            if matched_self == "Yes":
                success_count += 1

            writer.writerow([tid, prompt.strip(), matched_id, match_score, matched_self, explanation])

        except Exception as e:
            writer.writerow([tid, prompt.strip(), "Error", 0, "No", str(e)])

print(f"\n✅ Finished. {success_count} / {len(talents)} matched themselves with score ≥ 80%")
print(f"📄 Output saved to: {output_file}")
