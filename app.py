import streamlit as st
import spacy
import re
import logging
import traceback
import uuid
import time
import os
import psutil
import fitz # PyMuPDF

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Session State Initialization for Monitoring ---
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    logger.info(f"New session started for user: {st.session_state.user_id}")
if 'analysis_runs' not in st.session_state:
    st.session_state.analysis_runs = 0
if 'total_runtime' not in st.session_state:
    st.session_state.total_runtime = 0.0
if 'results' not in st.session_state:
    st.session_state.results = None

# This code assumes the 'en_core_web_md' model folder is in the same directory.
@st.cache_resource
def load_spacy_model():
    """
    Loads the SpaCy model with caching. Includes error handling.
    """
    try:
        start_time = time.time()
        model = spacy.load("en_core_web_md")
        load_duration = time.time() - start_time
        logger.info(f"Event: model_loaded | user_id: {st.session_state.user_id} | duration: {load_duration:.2f}s")
        return model
    except OSError as e:
        logger.error(f"Event: model_load_failed | error: 'OSError: {e}' | user_id: {st.session_state.user_id}")
        st.error("SpaCy model 'en_core_web_md' not found. Please ensure the model folder is in the same directory.")
        st.stop()
    except Exception as e:
        logger.error(f"Event: model_load_failed | error: 'Unexpected: {e}' | user_id: {st.session_state.user_id}\n{traceback.format_exc()}")
        st.error("An unexpected error occurred during model loading. Please try again or contact support.")
        st.stop()

nlp = load_spacy_model()

def normalize_degrees(text):
    """
    Normalizes common degree abbreviations and phrases into a standard format.
    """
    text = re.sub(r'master of science', 'master\'s degree', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmsc\b', 'master\'s degree', text, flags=re.IGNORECASE)
    text = re.sub(r'bachelor of science', 'bachelor\'s degree', text, flags=re.IGNORECASE)
    text = re.sub(r'\bbsc\b', 'bachelor\'s degree', text, flags=re.IGNORECASE)
    text = re.sub(r'ph\.?d\.?', 'doctorate degree', text, flags=re.IGNORECASE)
    return text

CUSTOM_STOP_WORDS = {"ideal", "candidate", "responsibilities", "requirements", "role", 
                     "experience", "work", "solution", "team", "project", "strong", 
                     "passion", "degree", "field", "innovations", "futuretech", "us",
                     "looking", "seeking", "join", "ability", "skill", "expertise"}

def extract_keywords_spacy(text):
    """
    Extracts potential keywords from text using SpaCy.
    """
    try:
        start_time = time.time()
        normalized_text = normalize_degrees(text)
        cleaned_text = re.sub(r'[^\w\s]', ' ', normalized_text)
        doc = nlp(cleaned_text.lower())
        keywords = set()

        for chunk in doc.noun_chunks:
            keywords.add(chunk.text)

        for token in doc:
            if token.pos_ in ["NOUN", "ADJ", "VERB"]:
                keywords.add(token.lemma_)

        filtered_keywords = {
            kw for kw in keywords
            if kw not in nlp.Defaults.stop_words and kw not in CUSTOM_STOP_WORDS and len(kw) > 1
        }
        duration = time.time() - start_time
        logger.debug(f"Event: keyword_extraction | duration: {duration:.4f}s | keywords_found: {len(filtered_keywords)}")
        return filtered_keywords
    except Exception as e:
        logger.error(f"Error during keyword extraction: {e}\n{traceback.format_exc()}")
        raise


def calculate_semantic_score(jd_keywords, resume_keywords, similarity_threshold=0.7):
    """
    Calculates a match score based on both exact keyword matches and semantic similarity.
    """
    if not jd_keywords:
        return 100, []

    start_time = time.time()
    matched_keywords = jd_keywords.intersection(resume_keywords)
    
    semantically_matched = set()
    for jd_kw in jd_keywords:
        if jd_kw not in matched_keywords:
            jd_token = nlp(jd_kw)
            if jd_token.has_vector:
                for res_kw in resume_keywords:
                    res_token = nlp(res_kw)
                    if res_token.has_vector:
                        try:
                            similarity = jd_token.similarity(res_token)
                            if similarity > similarity_threshold:
                                semantically_matched.add(jd_kw)
                                break
                        except Exception as e:
                            logger.warning(f"Could not calculate similarity between '{jd_kw}' and '{res_kw}': {e}")
            else:
                logger.debug(f"'{jd_kw}' has no vector, skipping semantic similarity check.")

    all_matched = matched_keywords.union(semantically_matched)
    
    score = (len(all_matched) / len(jd_keywords)) * 100
    truly_missing = jd_keywords - all_matched
    
    duration = time.time() - start_time
    logger.info(f"Event: semantic_score_calculated | score: {score:.2f}% | missing: {len(truly_missing)} | duration: {duration:.4f}s")
    return round(score, 2), truly_missing


def parse_resume_sections(resume_text):
    """
    Parses resume text into sections based on common headers.
    """
    sections = {}
    pattern = re.compile(r'^(?:summary|profile|experience|work history|education|skills|projects|certifications)\s*$', re.MULTILINE | re.IGNORECASE)
    
    matches = list(pattern.finditer(resume_text))
    
    if not matches:
        logger.debug("No distinct sections found, returning entire text as 'Uncategorized'.")
        return {"Uncategorized": resume_text}
        
    for i, match in enumerate(matches):
        section_name = match.group(0).strip().lower()
        start_index = match.end()
        end_index = matches[i+1].start() if i+1 < len(matches) else len(resume_text)
        
        section_content = resume_text[start_index:end_index].strip()
        sections[section_name] = section_content
    logger.debug(f"Parsed {len(sections)} resume sections.")
    return sections


def generate_contextual_suggestions(missing_keywords, sections):
    """
    Generates specific, section-based suggestions for truly missing keywords.
    """
    suggestions = {}
    
    skill_keywords = {"python", "aws", "machine learning", "pytorch", "gcp", "azure", "sql", "tableau", "docker", "r", "predictive modeling", "statistical modeling", "data analysis", "data analytics", "data visualization", "a/b testing", "api", "database", "cloud", "algorithms", "modeling"}
    experience_verbs = {"develop", "manage", "led", "implement", "create", "design", "build", "conduct", "deploy", "leverage", "optimize", "analyze", "integrate", "present", "contributed"}
    
    for keyword in missing_keywords:
        keyword_lower = keyword.lower()
        
        if any(sk in keyword_lower for sk in skill_keywords):
            target_section = "skills"
            if target_section in sections:
                suggestions.setdefault(target_section, []).append(f"💡 Consider adding '{keyword.title()}' to your **{target_section.capitalize()}** section.")
            else:
                suggestions.setdefault("General", []).append(f"💡 The job requires '{keyword.title()}'. You may want to add a dedicated **{target_section.capitalize()}** section to your resume.")
                
        elif any(ev in keyword_lower for ev in experience_verbs):
            target_section = "experience"
            if target_section in sections:
                suggestions.setdefault(target_section, []).append(f"💡 Add a bullet point in your **{target_section.capitalize()}** section that uses the term '{keyword.title()}', for example: '...'{keyword.title()}ed a scalable machine learning model...'")
            else:
                suggestions.setdefault("General", []).append(f"💡 The job description mentions '{keyword.title()}'. Consider adding an **{target_section.capitalize()}** section to showcase relevant projects or roles.")
        
        else:
            if "education" in sections and ("degree" in keyword_lower or "phd" in keyword_lower or "master" in keyword_lower or "bachelor" in keyword_lower):
                suggestions.setdefault("education", []).append(f"💡 Emphasize '{keyword.title()}' in your **Education** section, e.g., 'Master's degree with focus on {keyword.title()}'.")
            elif "summary" in sections or "profile" in sections:
                suggestions.setdefault("summary", []).append(f"💡 Integrate '{keyword.title()}' into your **Summary/Profile** to grab attention early.")
            else:
                suggestions.setdefault("General", []).append(f"💡 The job description mentions '{keyword.title()}'. Consider adding it to a relevant part of your resume.")
                
    logger.debug(f"Generated {len(suggestions)} types of contextual suggestions.")
    return suggestions

def get_text_from_pdf(uploaded_file):
    """
    Extracts text from a PDF file.
    """
    if uploaded_file is not None:
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            st.error("Error extracting text from the PDF file. Please ensure it's a valid PDF.")
    return ""

# --- Streamlit UI ---
st.set_page_config(page_title="AI Job Assistant PoC: Resume Matcher", layout="wide")
st.title("AI Job Assistant: Resume & Job Description Matcher (PoC)")
st.write("Upload your resume and enter a job description to see how well they align.")

# --- Monitoring Metrics in Sidebar ---
with st.sidebar:
    st.header("App Metrics")
    st.write(f"**Session ID:** `{st.session_state.user_id}`")
    st.write(f"**Analysis Runs:** `{st.session_state.analysis_runs}`")
    st.write(f"**Total Runtime:** `{st.session_state.total_runtime:.2f}`s")
    
    process = psutil.Process(os.getpid())
    st.write(f"**Memory Usage:** `{process.memory_info().rss / 1024 / 1024:.2f}` MB")
    
col1, col2 = st.columns(2)

with col1:
    st.header("Your Resume (PDF)")
    resume_file = st.file_uploader("Upload your resume as a PDF", type=["pdf"])

with col2:
    st.header("Job Description")
    jd_input = st.text_area("Paste the job description text here:", height=300)

if st.button("Analyze Resume vs. Job Description"):
    logger.info(f"Event: analyze_button_clicked | user_id: {st.session_state.user_id}")
    
    if resume_file is None or not jd_input:
        st.warning("Please upload a resume PDF and paste the job description text.")
        logger.warning(f"Event: input_missing | user_id: {st.session_state.user_id}")
    else:
        st.session_state.analysis_runs += 1
        try:
            with st.spinner("Analyzing..."):
                analysis_start_time = time.time()
                
                resume_text = get_text_from_pdf(resume_file)
                
                jd_keywords = extract_keywords_spacy(jd_input)
                resume_keywords = extract_keywords_spacy(resume_text)
                
                match_score, truly_missing_keywords = calculate_semantic_score(jd_keywords, resume_keywords)

                st.session_state.results = {
                    "match_score": match_score,
                    "missing_keywords": truly_missing_keywords,
                    "jd_keywords": jd_keywords,
                    "resume_keywords": resume_keywords,
                    "initial_resume_text": resume_text
                }
                
                analysis_duration = time.time() - analysis_start_time
                st.session_state.total_runtime += analysis_duration
                logger.info(f"Event: analysis_completed | user_id: {st.session_state.user_id} | duration: {analysis_duration:.2f}s | jd_keywords: {len(jd_keywords)} | resume_keywords: {len(resume_keywords)}")
        except Exception as e:
            logger.error(f"Event: analysis_failed | user_id: {st.session_state.user_id} | error: {e}\n{traceback.format_exc()}")
            st.error("An error occurred during analysis. Please check your inputs or try again later.")

# Display Results Section
if st.session_state.results:
    results = st.session_state.results

    st.markdown("---")
    st.subheader("Match Score 🎯")
    st.markdown(f"**Your resume matches {results['match_score']}% of the keywords in the job description.**")
    st.progress(results['match_score'] / 100)
    
    if results['match_score'] >= 80:
        st.success("This is a great match! Your resume is highly aligned with the job description.")
    elif results['match_score'] >= 50:
        st.info("A good match. Review the suggestions below to further improve your resume.")
    else:
        st.warning("The match is low. You may need to significantly tailor your resume to this role.")
    
    st.markdown("---")

    st.subheader("Extracted Job Description Keywords:")
    st.write(", ".join(sorted(list(results['jd_keywords']))))

    st.subheader("Extracted Resume Keywords:")
    st.write(", ".join(sorted(list(results['resume_keywords']))))

    st.subheader("Keywords from Job Description *NOT* found in your Resume:")
    if results['missing_keywords']:
        st.warning(", ".join(sorted(list(results['missing_keywords']))))
    else:
        st.success("Great! Your resume seems to cover all the key terms from the job description.")

    st.subheader("Contextual Suggestions:")
    if results['missing_keywords']:
        resume_sections = parse_resume_sections(results['initial_resume_text'])
        contextual_suggestions = generate_contextual_suggestions(results['missing_keywords'], resume_sections)

        for section, suggestions_list in contextual_suggestions.items():
            st.write(f"**Suggestions for your '{section.capitalize()}' section:**")
            
            for suggestion in suggestions_list:
                st.info(suggestion)
    else:
        st.success("Your resume is well-aligned with the job description. No contextual suggestions needed.")
        
st.markdown("---")
st.markdown("### How this PoC works:")
st.markdown("""
1.  It pre-processes text by **normalizing common degree phrases** like "Master of Science" to "Master's degree."
2.  It uses **SpaCy** and **word embeddings** to extract and compare keywords from both texts. The model is now **cached** for faster subsequent runs.
3.  It calculates a **semantic match score** based on the percentage of job description keywords found in your resume, considering both direct matches and semantic similarity.
4.  It identifies which keywords from the **Job Description** are truly missing from your **Resume**.
5.  It provides more detailed **contextual suggestions**, guiding you on where to place keywords in your resume.
""")