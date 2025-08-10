import streamlit as st
import spacy
import re
import logging
import traceback
import uuid
import time
import os
import psutil
import fitz  # PyMuPDF

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
    st.session_state.user_id = str(uuid.uuid4())
    logger.info(f"New session started for user: {st.session_state.user_id}")
if 'analysis_runs' not in st.session_state:
    st.session_state.analysis_runs = 0
if 'total_runtime' not in st.session_state:
    st.session_state.total_runtime = 0.0
if 'results' not in st.session_state:
    st.session_state.results = None

# =========================
# --- Load SpaCy Model ---
# =========================
@st.cache_resource
def load_spacy_model():
    try:
        start_time = time.time()
        model = spacy.load("en_core_web_md")
        logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
        return model
    except OSError as e:
        logger.error(f"OSError loading model: {e}")
        st.error("SpaCy model 'en_core_web_md' not found. Please install it.")
        st.stop()
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}\n{traceback.format_exc()}")
        st.error("Unexpected error loading SpaCy model.")
        st.stop()

nlp = load_spacy_model()

# =========================
# --- Utility Functions ---
# =========================
def normalize_degrees(text: str) -> str:
    text = re.sub(r'master of science', "master's degree", text, flags=re.IGNORECASE)
    text = re.sub(r'\bmsc\b', "master's degree", text, flags=re.IGNORECASE)
    text = re.sub(r'bachelor of science', "bachelor's degree", text, flags=re.IGNORECASE)
    text = re.sub(r'\bbsc\b', "bachelor's degree", text, flags=re.IGNORECASE)
    text = re.sub(r'ph\.?d\.?', "doctorate degree", text, flags=re.IGNORECASE)
    return text

@st.cache_data
def extract_keywords_spacy(text: str) -> set:
    try:
        start_time = time.time()
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

        filtered = {
            kw for kw in keywords
            if kw not in nlp.Defaults.stop_words
            and kw not in CUSTOM_STOP_WORDS
            and len(kw) > 1
        }

        logger.debug(f"Extracted {len(filtered)} keywords in {time.time() - start_time:.4f}s")
        return filtered

    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}\n{traceback.format_exc()}")
        return set()

def calculate_semantic_score(jd_keywords: set, resume_keywords: set, similarity_threshold=0.7):
    if not jd_keywords:
        return 100, []

    start_time = time.time()
    matched_keywords = jd_keywords & resume_keywords

    # Precompute vectors
    jd_vectors = {kw: nlp(kw) for kw in jd_keywords if nlp(kw).has_vector}
    res_vectors = {kw: nlp(kw) for kw in resume_keywords if nlp(kw).has_vector}

    semantically_matched = set()
    for jd_kw, jd_doc in jd_vectors.items():
        if jd_kw not in matched_keywords:
            for res_doc in res_vectors.values():
                try:
                    if jd_doc.similarity(res_doc) > similarity_threshold:
                        semantically_matched.add(jd_kw)
                        break
                except Exception as e:
                    logger.debug(f"Similarity calc failed for {jd_kw}: {e}")

    all_matched = matched_keywords | semantically_matched
    score = (len(all_matched) / len(jd_keywords)) * 100
    truly_missing = jd_keywords - all_matched

    logger.info(f"Semantic score {score:.2f}% | Missing {len(truly_missing)} | Took {time.time() - start_time:.2f}s")
    return round(score, 2), truly_missing

def parse_resume_sections(resume_text: str) -> dict:
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
    suggestions = {}
    for keyword in missing_keywords:
        keyword_lower = keyword.lower()

        if any(sk in keyword_lower for sk in SKILL_KEYWORDS):
            target = "skills"
            if target in sections:
                suggestions.setdefault(target, []).append(
                    f"💡 Consider adding '{keyword.title()}' to your **{target.capitalize()}** section.")
            else:
                suggestions.setdefault("General", []).append(
                    f"💡 Add a **{target.capitalize()}** section mentioning '{keyword.title()}'.")
        elif any(ev in keyword_lower for ev in EXPERIENCE_VERBS):
            target = "experience"
            if target in sections:
                suggestions.setdefault(target, []).append(
                    f"💡 Include '{keyword.title()}' in a bullet point under **{target.capitalize()}**.")
            else:
                suggestions.setdefault("General", []).append(
                    f"💡 Add an **{target.capitalize()}** section for '{keyword.title()}'.")
        else:
            if "education" in sections and any(x in keyword_lower for x in ["degree", "phd", "master", "bachelor"]):
                suggestions.setdefault("education", []).append(
                    f"💡 Mention '{keyword.title()}' in your **Education** section.")
            elif "summary" in sections or "profile" in sections:
                suggestions.setdefault("summary", []).append(
                    f"💡 Add '{keyword.title()}' to your **Summary/Profile**.")
            else:
                suggestions.setdefault("General", []).append(
                    f"💡 Consider adding '{keyword.title()}' to a relevant section.")
    return suggestions

def get_text_from_pdf(uploaded_file) -> str:
    if uploaded_file:
        try:
            doc = fitz.open(stream=uploaded_file, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            st.error("Could not read the PDF.")
    return ""

def generate_alternative_phrasing(missing_keywords: set) -> dict:
    suggestions = {}
    for keyword in missing_keywords:
        for key, alternatives in ALTERNATIVE_PHRASING_MAP.items():
            if key in keyword.lower():
                suggestions.setdefault(key, set()).update(alternatives)
    return suggestions
