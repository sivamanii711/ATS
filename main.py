import os
import json
from flask import Flask, request, jsonify, render_template
from google import genai
import PyPDF2

# ==============================
# CONFIG
# ==============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Replace with your actual API Key
client = genai.Client(api_key="YOUR_API_KEY")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# PDF PARSING
# ==============================
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text

# ==============================
# ATS MATCHING (CONNECTED LOGIC)
# ==============================
def get_ats_analysis(resume_text, jd_text):
    """
    Combines parsing and matching into one structured LLM call
    to ensure the frontend receives valid JSON data.
    """
    prompt = f"""
    You are an expert Applicant Tracking System (ATS). 
    Analyze the following Resume against the Job Description.
    
    RESUME:
    {resume_text}
    
    JOB DESCRIPTION:
    {jd_text}
    
    Return a valid JSON object with the following keys:
    - match_percentage (integer 0-100)
    - matching_skills (list of strings)
    - missing_skills (list of strings)
    - summary (a 2-3 sentence overview of the candidate's fit)
    - suggestions (list of 3 specific improvement points)
    """

    # We use gemini-2.5-flash-preview-09-2025 and request a JSON response
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-09-2025",
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )
    
    return json.loads(response.text)

# ==============================
# ROUTES
# ==============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "Resume PDF is required"}), 400

    resume_file = request.files["resume"]
    jd_text = request.form.get("job_description")

    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400

    try:
        # 1. Save PDF temporarily
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
        resume_file.save(pdf_path)

        # 2. Extract resume text
        resume_text = extract_text_from_pdf(pdf_path)
        
        if not resume_text.strip():
            return jsonify({"error": "Could not extract text from PDF"}), 400

        # 3. Get Structured Analysis from Gemini
        analysis_result = get_ats_analysis(resume_text, jd_text)

        # 4. Clean up file after processing
        os.remove(pdf_path)

        return jsonify(analysis_result)

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": "An error occurred during analysis"}), 500

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True, port=8080)