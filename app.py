"""
AI Job Assistant PoC - Resume Matcher
-------------------------------------
This Streamlit application compares a candidate's resume with a job description
to determine how well they match. It extracts keywords, calculates a match score,
identifies missing skills, and gives contextual suggestions for improvement.

Mobile-friendly features:
- Centered layout for smaller screens
- Full-width buttons for touch devices
- Expanders to hide long lists
- Color-coded badge for instant match score reading
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

# =========================
# --- Session State ---
# =========================
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())  # Unique session identifier
if 'analysis_runs' not in st.session_state:
    st.session_state.analysis_runs = 0  # Number of analyses run in this session
if 'total_runtime' not in st.session_state:
    st.session_state.total_runtime = 0.0  # Accumulated processing time
if 'results' not in st.session_state:
    st.session_state.results = None  # Store last results

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
st.write("Upload your resume and job description to check alignment.")

# Sidebar for session info
with st.sidebar:
    st.header("Metrics")
    st.markdown(f"**Session ID:** `{st.session_state.user_id}`")
    st.markdown(f"**Runs:** `{st.session_state.analysis_runs}`")
    st.markdown(f"**Total Runtime:** `{st.session_state.total_runtime:.2f}`s")
    process = psutil.Process(os.getpid())
    st.markdown(f"**Memory:** `{process.memory_info().rss / 1024 / 1024:.2f}` MB")

# File upload & job description text
resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])
jd_input = st.text_area("Job Description", height=200, placeholder="Paste job description here...")

# Run analysis button
if st.button("Analyze Match", use_container_width=True):
    if resume_file and jd_input.strip():
        st.session_state.analysis_runs += 1
        start_time = time.time()

        # Process input
        resume_text = get_text_from_pdf(resume_file)
        jd_keywords = extract_keywords_spacy(jd_input)
        resume_keywords = extract_keywords_spacy(resume_text)

        # Compare and generate results
        match_score, missing_keywords = calculate_semantic_score(jd_keywords, resume_keywords)
        alternative_phrasing = generate_alternative_phrasing(missing_keywords)

        # Save in session
        st.session_state.results = {
            "match_score": match_score,
            "missing_keywords": missing_keywords,
            "jd_keywords": jd_keywords,
            "resume_keywords": resume_keywords,
            "resume_text": resume_text,
            "alternative_phrasing": alternative_phrasing
        }
        st.session_state.total_runtime += time.time() - start_time
    else:
        st.warning("Please upload a PDF resume and paste a job description.")

# =========================
# --- Results Section ---
# =========================
if st.session_state.results:
    results = st.session_state.results

    # Determine badge color based on score
    if results['match_score'] >= 80:
        badge_color = "#4CAF50"  # Green
    elif results['match_score'] >= 50:
        badge_color = "#FFC107"  # Amber
    else:
        badge_color = "#F44336"  # Red

    # Score display with badge
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

    # Expanders for details
    with st.expander("Job Description Keywords"):
        st.write(", ".join(sorted(results['jd_keywords'])))

    with st.expander("Resume Keywords"):
        st.write(", ".join(sorted(results['resume_keywords'])))

    with st.expander("Missing Keywords"):
        if results['missing_keywords']:
            st.warning(", ".join(sorted(results['missing_keywords'])))
        else:
            st.success("All keywords covered!")

    # Contextual suggestions
    if results['missing_keywords']:
        with st.expander("Contextual Suggestions"):
            sections = parse_resume_sections(results['resume_text'])
            suggestions = generate_contextual_suggestions
