"""
AI Job Assistant PoC - Resume Matcher with Job Search
-----------------------------------------------------
This Streamlit application helps a candidate find a matching job.
It allows users to upload their resume, browse a list of sample job descriptions,
filter them, and then get a detailed analysis of how well their resume matches the selected job.

MVP Idea 2: Add basic job search and filtering (manual application).
"""

import streamlit as st
import spacy
import re
import logging
import traceback
import uuid
import time
import os
import psutil
import fitz  # PyMuPDF for PDF parsing

# =========================
# --- Configure Logging ---
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# --- Constants ---
# =========================
CUSTOM_STOP_WORDS = {
    "ideal", "candidate", "responsibilities", "requirements", "role",
    "experience", "work", "solution", "team", "project", "strong",
    "passion", "degree", "field", "innovations", "futuretech", "us",
    "looking", "seeking", "join", "ability", "skill", "expertise"
}

SKILL_KEYWORDS = {
    "python", "aws", "machine learning", "pytorch", "gcp", "azure", "sql", "tableau",
    "docker", "r", "predictive modeling", "statistical modeling", "data analysis",
    "data analytics", "data visualization", "a/b testing", "api", "database",
    "cloud", "algorithms", "modeling"
}

EXPERIENCE_VERBS = {
    "develop", "manage", "led", "implement", "create", "design", "build", "conduct",
    "deploy", "leverage", "optimize", "analyze", "integrate", "present", "contributed"
}

# Alternative phrases to suggest when keywords are missing
ALTERNATIVE_PHRASING_MAP = {
    "tensorflow": ["deep learning frameworks", "neural networks", "machine learning libraries"],
    "pytorch": ["deep learning frameworks", "neural networks"],
    "sql": ["database management", "relational databases", "data querying"],
    "aws": ["cloud services", "cloud computing", "amazon web services"],
    "nlp": ["natural language processing", "text analysis", "text mining"],
    "docker": ["containerization", "devops tools", "microservices"],
    "predictive modeling": ["statistical analysis", "forecasting"],
    "data visualization": ["tableau", "power bi", "matplotlib"],
    "algorithms": ["data structures", "problem-solving skills"]
}

# Hard-coded list of sample job descriptions for MVP
JOB_LIST = [
    {
        "id": 1,
        "title": "Senior Data Scientist",
        "company": "Tech Innovations Inc.",
        "location": "San Francisco, CA",
        "description": """
        Overview:
        We are seeking an experienced and highly motivated Senior Data Scientist to lead our data-driven initiatives. The ideal candidate will be a strategic thinker with a proven track record of developing and deploying advanced machine learning models that deliver significant business value. In this senior role, you will not only be a technical expert but also a mentor to junior team members, helping to shape our data science roadmap and drive best practices.

        Key Responsibilities:
        - Lead the entire lifecycle of data science projects, from problem formulation and data collection to model deployment and monitoring.
        - Design, build, and deploy sophisticated machine learning, deep learning, and statistical models to solve complex business problems.
        - Analyze large, unstructured datasets to extract actionable insights and identify key trends.
        - Collaborate with business stakeholders, product managers, and engineering teams to translate business needs into technical requirements.
        - Mentor and guide junior data scientists, fostering a culture of continuous learning and technical excellence.
        - Communicate complex findings and recommendations clearly and concisely to both technical and non-technical audiences.
        - Evaluate and recommend new technologies and methodologies to enhance our data science capabilities.

        Required Qualifications:
        - Master's or Ph.D. in Computer Science, Statistics, Mathematics, or a related quantitative field.
        - Minimum of 5+ years of hands-on experience as a Data Scientist or in a similar role.
        - Expert-level proficiency in Python and its data science ecosystem (NumPy, pandas, scikit-learn).
        - Deep expertise with machine learning frameworks like TensorFlow or PyTorch.
        - Strong knowledge of SQL for data extraction and manipulation.
        - Proven experience with cloud platforms such as AWS, GCP, or Azure.
        - Excellent problem-solving skills and a strong understanding of statistical modeling and data mining techniques.
        """,
        "url": "https://www.example-job-1.com"
    },
    {
        "id": 2,
        "title": "Junior Data Analyst",
        "company": "Global Analytics Co.",
        "location": "New York, NY",
        "description": """
        Overview:
        Global Analytics Co. is looking for a Junior Data Analyst to join our growing team. You will be responsible for collecting, cleaning, and analyzing data to support our business teams. This is an excellent opportunity for an entry-level professional to gain hands-on experience in a fast-paced environment.

        Key Responsibilities:
        - Collect data from various sources and maintain databases.
        - Clean and preprocess data to ensure accuracy and consistency.
        - Create reports and dashboards using Tableau to visualize data insights.
        - Assist senior analysts with ad-hoc data analysis requests.
        - Communicate findings to team members.

        Required Qualifications:
        - Bachelor's degree in a quantitative field (e.g., Mathematics, Economics, Statistics).
        - Proficiency in SQL and Excel.
        - Experience with data visualization tools like Tableau.
        - Basic knowledge of statistical analysis.
        - Strong attention to detail and communication skills.
        """,
        "url": "https://www.example-job-2.com"
    },
    {
        "id": 3,
        "title": "Machine Learning Engineer",
        "company": "NextGen AI",
        "location": "Seattle, WA",
        "description": """
        Overview:
        We are seeking a talented Machine Learning Engineer to design and implement robust, scalable AI systems. The ideal candidate has strong software engineering skills and a deep understanding of machine learning principles. You will be instrumental in taking our models from research to production.

        Key Responsibilities:
        - Develop and deploy machine learning models at scale.
        - Build and maintain data pipelines for model training and inference.
        - Work with cloud platforms (AWS, Azure) and containerization technologies (Docker, Kubernetes).
        - Collaborate with data scientists to optimize model performance.
        - Implement CI/CD pipelines for ML models.

        Required Qualifications:
        - Bachelor's or Master's degree in Computer Science or a related field.
        - Strong programming skills in Python.
        - 3+ years of experience in a machine learning or software engineering role.
        - Experience with MLOps and productionizing ML models.
        - Familiarity with deep learning frameworks like TensorFlow or PyTorch.
        """,
        "url": "https://www.example-job-3.com"
    }
]


# =========================
# --- Session State ---
# =========================
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'analysis_runs' not in st.session_state:
    st.session_state.analysis_runs = 0
if 'total_runtime' not in st.session_state:
    st.session_state.total_runtime = 0.0
if 'selected_job' not in st.session_state:
    st.session_state.selected_job = None
if 'results' not in st.session_state:
    st.session_state.results = None

# =========================
# --- Load SpaCy Model ---
# =========================
@st.cache_resource
def load_spacy_model():
    """
    Load the SpaCy medium English model.
    Caching prevents reloading on each app refresh.
    """
    try:
        start_time = time.time()
        model = spacy.load("en_core_web_md")
        logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
        return model
    except OSError:
        st.error("SpaCy model 'en_core_web_md' not found. Please install it with `python -m spacy download en_core_web_md`.")
        st.stop()
    except Exception:
        st.error("Unexpected error loading SpaCy model.")
        st.stop()

nlp = load_spacy_model()

# =========================
# --- Text Preprocessing ---
# =========================
def normalize_degrees(text: str) -> str:
    """
    Normalize common degree abbreviations in text for better keyword matching.
    """
    text = re.sub(r'master of science', "master's degree", text, flags=re.IGNORECASE)
    text = re.sub(r'\bmsc\b', "master's degree", text, flags=re.IGNORECASE)
    text = re.sub(r'bachelor of science', "bachelor's degree", text, flags=re.IGNORECASE)
    text = re.sub(r'\bbsc\b', "bachelor's degree", text, flags=re.IGNORECASE)
    text = re.sub(r'ph\.?d\.?', "doctorate degree", text, flags=re.IGNORECASE)
    return text

@st.cache_data
def extract_keywords_spacy(text: str) -> set:
    """
    Extract keywords using SpaCy noun chunks + lemmas, removing stop words and custom noise terms.
    """
    try:
        text = normalize_degrees(text.lower())
        text = re.sub(r'[^\w\s]', ' ', text)
        doc = nlp(text)

        keywords, chunk_indices = set(), set()
        for chunk in doc.noun_chunks:
            keywords.add(chunk.text)
            chunk_indices.update(t.i for t in chunk)

        for token in doc:
            if token.i not in chunk_indices and token.pos_ in {"NOUN", "ADJ", "VERB"}:
                keywords.add(token.lemma_)

        return {
            kw for kw in keywords
            if kw not in nlp.Defaults.stop_words
            and kw not in CUSTOM_STOP_WORDS
            and len(kw) > 1
        }
    except Exception:
        return set()

def calculate_semantic_score(jd_keywords: set, resume_keywords: set, similarity_threshold=0.7):
    """
    Compare job description keywords with resume keywords.
    Includes direct matches and semantic matches above the given threshold.
    """
    if not jd_keywords:
        return 100, []

    matched_keywords = jd_keywords & resume_keywords

    jd_vectors = {kw: nlp(kw) for kw in jd_keywords if nlp(kw).has_vector}
    res_vectors = {kw: nlp(kw) for kw in resume_keywords if nlp(kw).has_vector}

    semantically_matched = set()
    for jd_kw, jd_doc in jd_vectors.items():
        if jd_kw not in matched_keywords:
            for res_doc in res_vectors.values():
                if jd_doc.similarity(res_doc) > similarity_threshold:
                    semantically_matched.add(jd_kw)
                    break

    all_matched = matched_keywords | semantically_matched
    score = (len(all_matched) / len(jd_keywords)) * 100
    return round(score, 2), jd_keywords - all_matched

def parse_resume_sections(resume_text: str) -> dict:
    """
    Split resume into sections based on common headings.
    """
    sections = {}
    pattern = re.compile(r'^(?:summary|profile|experience|work history|education|skills|projects|certifications)\s*$',
                         re.MULTILINE | re.IGNORECASE)
    matches = list(pattern.finditer(resume_text))
    if not matches:
        return {"Uncategorized": resume_text}

    for i, match in enumerate(matches):
        section_name = match.group(0).strip().lower()
        start_index = match.end()
        end_index = matches[i + 1].start() if i + 1 < len(matches) else len(resume_text)
        sections[section_name] = resume_text[start_index:end_index].strip()
    return sections

def generate_contextual_suggestions(missing_keywords: set, sections: dict) -> dict:
    """
    Suggest where missing keywords could be added in the resume.
    """
    suggestions = {}
    for keyword in missing_keywords:
        kw_lower = keyword.lower()
        if any(sk in kw_lower for sk in SKILL_KEYWORDS):
            target = "skills"
            if target in sections:
                suggestions.setdefault(target, []).append(f"💡 Add '{keyword.title()}' to your **Skills** section.")
            else:
                suggestions.setdefault("General", []).append(f"💡 Create a **Skills** section with '{keyword.title()}'.")
        elif any(ev in kw_lower for ev in EXPERIENCE_VERBS):
            target = "experience"
            if target in sections:
                suggestions.setdefault(target, []).append(f"💡 Mention '{keyword.title()}' in **Experience**.")
            else:
                suggestions.setdefault("General", []).append(f"💡 Add an **Experience** section mentioning '{keyword.title()}'.")
        else:
            if "education" in sections and any(x in kw_lower for x in ["degree", "phd", "master", "bachelor"]):
                suggestions.setdefault("education", []).append(f"💡 Highlight '{keyword.title()}' in **Education**.")
            elif "summary" in sections or "profile" in sections:
                suggestions.setdefault("summary", []).append(f"💡 Add '{keyword.title()}' to **Summary/Profile**.")
            else:
                suggestions.setdefault("General", []).append(f"💡 Include '{keyword.title()}' in a relevant section.")
    return suggestions

def get_text_from_pdf(uploaded_file) -> str:
    """
    Extract raw text from uploaded PDF file.
    """
    if uploaded_file:
        try:
            doc = fitz.open(stream=uploaded_file, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        except Exception:
            st.error("Could not read the PDF.")
    return ""

def generate_alternative_phrasing(missing_keywords: set) -> dict:
    """
    Provide alternative keyword suggestions for missing terms.
    """
    suggestions = {}
    for keyword in missing_keywords:
        for key, alternatives in ALTERNATIVE_PHRASING_MAP.items():
            if key in keyword.lower():
                suggestions.setdefault(key, set()).update(alternatives)
    return suggestions

# =========================
# --- Streamlit UI ---
# =========================
st.set_page_config(page_title="AI Job Assistant PoC", layout="centered")
st.title("AI Job Assistant 📄🤝💼")
st.write("Upload your resume, find a job, and get a match analysis.")

# Sidebar for session info
with st.sidebar:
    st.header("Metrics")
    st.markdown(f"**Session ID:** `{st.session_state.user_id}`")
    st.markdown(f"**Runs:** `{st.session_state.analysis_runs}`")
    st.markdown(f"**Total Runtime:** `{st.session_state.total_runtime:.2f}`s")
    process = psutil.Process(os.getpid())
    st.markdown(f"**Memory:** `{process.memory_info().rss / 1024 / 1024:.2f}` MB")

# Main columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Your Resume (PDF)")
    resume_file = st.file_uploader("Upload your resume", type=["pdf"])

with col2:
    st.header("Job Search")
    search_query = st.text_input("Filter jobs by title or company:")
    st.markdown("---")

    filtered_jobs = [
        job for job in JOB_LIST
        if search_query.lower() in job['title'].lower() or search_query.lower() in job['company'].lower()
    ]

    if filtered_jobs:
        for job in filtered_jobs:
            if st.button(f"**{job['title']}** at {job['company']}", key=f"job-{job['id']}", use_container_width=True):
                st.session_state.selected_job = job
                st.session_state.results = None  # Clear previous results
    else:
        st.info("No jobs match your search query.")

# Display selected job and analysis button
if st.session_state.selected_job:
    job = st.session_state.selected_job
    st.markdown("---")
    st.header(job['title'])
    st.subheader(job['company'])
    st.write(f"📍 {job['location']}")
    
    with st.expander("View Full Job Description"):
        st.markdown(job['description'])
        st.markdown(f"[Apply Now]({job['url']})", unsafe_allow_html=True)
    
    # Analyze button is now context-aware of the selected job
    if st.button("Analyze Match with Resume", use_container_width=True):
        if resume_file:
            st.session_state.analysis_runs += 1
            start_time = time.time()
            
            # --- Analysis logic ---
            try:
                resume_text = get_text_from_pdf(resume_file)
                jd_keywords = extract_keywords_spacy(job['description'])
                resume_keywords = extract_keywords_spacy(resume_text)
                
                match_score, missing_keywords = calculate_semantic_score(jd_keywords, resume_keywords)
                alternative_phrasing = generate_alternative_phrasing(missing_keywords)
                
                st.session_state.results = {
                    "match_score": match_score,
                    "missing_keywords": missing_keywords,
                    "jd_keywords": jd_keywords,
                    "resume_keywords": resume_keywords,
                    "resume_text": resume_text,
                    "alternative_phrasing": alternative_phrasing
                }
                st.session_state.total_runtime += time.time() - start_time
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                logger.error(f"Error during analysis: {traceback.format_exc()}")
        else:
            st.warning("Please upload your resume as a PDF file first.")

# =========================
# --- Results Section ---
# =========================
if st.session_state.results:
    results = st.session_state.results
    st.markdown("---")
    
    if results['match_score'] >= 80:
        badge_color = "#4CAF50"  # Green
    elif results['match_score'] >= 50:
        badge_color = "#FFC107"  # Amber
    else:
        badge_color = "#F44336"  # Red

    st.markdown(
        f"""
        <div style='display:flex; align-items:center; gap:10px;'>
            <h3 style='margin:0;'>Match Score 🎯: {results['match_score']}%</h3>
            <span style='background-color:{badge_color}; color:white; padding:5px 10px; border-radius:12px; font-weight:bold;'>
                {('High' if badge_color=="#4CAF50" else 'Medium' if badge_color=="#FFC107" else 'Low')}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(results['match_score'] / 100)

    with st.expander("Job Description Keywords"):
        st.write(", ".join(sorted(results['jd_keywords'])))

    with st.expander("Resume Keywords"):
        st.write(", ".join(sorted(results['resume_keywords'])))

    with st.expander("Missing Keywords"):
        if results['missing_keywords']:
            st.warning(", ".join(sorted(results['missing_keywords'])))
        else:
            st.success("All keywords covered!")

    with st.expander("Contextual Suggestions"):
        if results['missing_keywords']:
            sections = parse_resume_sections(results['resume_text'])
            suggestions = generate_contextual_suggestions(results['missing_keywords'], sections)
            
            if suggestions:
                for section, suggestions_list in suggestions.items():
                    st.markdown(f"**Suggestions for your '{section}' section:**")
                    for suggestion in suggestions_list:
                        st.info(suggestion)
            else:
                st.info("No specific contextual suggestions could be generated.")
        else:
            st.success("Your resume is well-aligned with the job description. No contextual suggestions needed.")

    with st.expander("Alternative Phrasing"):
        if results['alternative_phrasing']:
            st.write("Consider including these related skills or alternative phrases:")
            for missing_skill, alternatives in results['alternative_phrasing'].items():
                st.markdown(f"**{missing_skill.title()}:** {', '.join(sorted(list(alternatives)))}")
        else:
            st.info("No alternative phrasing suggestions at this time.")

    st.markdown("---")
    st.info(f"You've chosen to apply for: **{st.session_state.selected_job['title']}** at **{st.session_state.selected_job['company']}**")
    st.markdown(f"[Apply to this job now]({st.session_state.selected_job['url']})", unsafe_allow_html=True)