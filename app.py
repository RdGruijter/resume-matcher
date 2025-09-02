"""
AI Job Assistant PoC - Resume Matcher with Live Job Search
---------------------------------------------------------
This Streamlit application helps a candidate find a matching job.
It allows users to upload their resume, browse live job listings,
and get a detailed match analysis powered by Llama 3.1.
"""

import streamlit as st
import re
import logging
import traceback
import uuid
import time
import os
import psutil
import fitz  # PyMuPDF for PDF parsing
import requests # For making API requests
import json   # For loading industry keywords from JSON

# =========================
# --- Configure Logging ---
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# --- Constants and Data Loading ---
# =========================
@st.cache_data
def load_industry_keywords():
    """Loads industry keywords from a JSON file."""
    try:
        with open("data/keywords.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("keywords.json not found in the 'data/' directory. Please create it.")
        return {}
    except json.JSONDecodeError:
        st.error("Error decoding keywords.json. Please check its format.")
        return {}

INDUSTRY_KEYWORDS = load_industry_keywords()

# =========================
# --- API Credentials ---
# =========================
RAPIDAPI_KEY = st.secrets["RAPIDAPI_KEY"]
RAPIDAPI_HOST = st.secrets["RAPIDAPI_HOST"]
LLAMA3_RAPIDAPI_KEY = st.secrets["LLAMA3_RAPIDAPI_KEY"]
LLAMA3_RAPIDAPI_HOST = st.secrets["LLAMA3_RAPIDAPI_HOST"]

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
if 'llm_analysis_results' not in st.session_state:
    st.session_state.llm_analysis_results = None
if 'jobs' not in st.session_state:
    st.session_state.jobs = []

# =========================
# --- API & Data Fetching Functions ---
# =========================
def fetch_jobs_from_api(query, country_code):
    """Fetches job listings from the JSearch API on RapidAPI."""
    base_url = f"https://{RAPIDAPI_HOST}/search"
    
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    
    params = {
        "query": query,
        "country": country_code,
        "num_pages": 1
    }

    try:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        jobs = []
        for job_data in data.get('data', []):
            jobs.append({
                "id": job_data.get('job_id'),
                "title": job_data.get('job_title'),
                "company": job_data.get('employer_name', 'N/A'),
                "location": job_data.get('job_country', 'N/A'),
                "description": job_data.get('job_description'),
                "url": job_data.get('job_apply_link')
            })
        return jobs
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching jobs from RapidAPI: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while processing job data: {e}")
        return []

def analyze_match_llm(resume_text, job_description):
    """Analyzes the resume and job description using Llama 3.1 via RapidAPI."""
    url = f"https://{LLAMA3_RAPIDAPI_HOST}/v1/chat/completions"

    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": LLAMA3_RAPIDAPI_KEY,
        "X-RapidAPI-Host": LLAMA3_RAPIDAPI_HOST
    }
    
    prompt_content = f"""
    You are an expert resume analyzer. Your task is to compare a candidate's resume with a job description and provide a comprehensive, structured analysis.

    Job Description:
    {job_description}

    Candidate's Resume:
    {resume_text}

    Analyze the job description and resume and provide the following information in a structured text format:
    1.  **Match Score:** A single number from 1 to 100 representing the overall match percentage.
    2.  **Top Strengths:** A list of 3-5 key skills or experiences from the resume that are highly relevant to the job.
    3.  **Areas for Improvement:** A list of 3-5 specific skills, experiences, or keywords from the job description that are missing or under-represented in the resume.
    4.  **Summary:** A concise paragraph summarizing why the candidate is a good fit and how they can improve their resume for this specific role.

    Format your response exactly as follows, using bold for headings:
    **Match Score:** [score]%
    **Top Strengths:**
    - [Strength 1]
    - [Strength 2]
    - [Strength 3]
    **Areas for Improvement:**
    - [Missing Skill 1]
    - [Missing Skill 2]
    - [Missing Skill 3]
    **Summary:** [Your concise summary paragraph here.]
    """
    
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        "max_tokens": 500,
        "temperature": 0.3
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Llama 3.1 API: {e}")
        logger.error(f"Llama 3.1 API Error: {traceback.format_exc()}")
        return "Sorry, an error occurred while performing the analysis."
    except Exception as e:
        st.error(f"An unexpected error occurred while processing LLM data: {e}")
        logger.error(f"LLM Data Processing Error: {traceback.format_exc()}")
        return "Sorry, an unexpected error occurred."

# =========================
# --- Text Preprocessing ---
# =========================
def get_text_from_pdf(uploaded_file) -> str:
    """Extracts raw text from an uploaded PDF file."""
    if uploaded_file:
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            st.error(f"Could not read the PDF: {e}")
            logger.error(f"PDF Reading Error: {traceback.format_exc()}")
    return ""

# =========================
# --- Streamlit UI ---
# =========================
st.set_page_config(page_title="AI Job Assistant PoC", layout="centered")
st.title("AI Job Assistant 📄🤝💼")
st.write("Upload your resume, search for a job, and get a match analysis powered by Llama 3.1.")

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
    
    countries = {
        "United States": "us", "United Kingdom": "gb", "Canada": "ca",
        "Australia": "au", "Germany": "de", "France": "fr",
        "India": "in", "Mexico": "mx", "Netherlands": "nl"
    }
    selected_country = st.selectbox("Select Country", list(countries.keys()), key="country_select")
    country_code = countries[selected_country]

    search_query = st.text_input("Enter job title or keyword (e.g., 'data scientist'):", key="search_input")
    
    if st.button("Search Jobs", use_container_width=True) and search_query:
        with st.status("Fetching jobs...", expanded=True) as status:
            st.session_state.jobs = fetch_jobs_from_api(search_query, country_code)
            st.session_state.selected_job = None
            st.session_state.llm_analysis_results = None
            
            if st.session_state.jobs:
                status.update(label="Job listings fetched!", state="complete", expanded=False)
            else:
                status.update(label="No jobs found!", state="complete", expanded=False)
                st.write(f"😔 No jobs found for '{search_query}' in {selected_country}. Please try a different search or country.")

    st.markdown("---")

    if st.session_state.jobs:
        job_titles = [f"{job['title']} at {job['company']}" for job in st.session_state.jobs]
        selected_title = st.selectbox(
            "Select a Job to Analyze",
            job_titles,
            key="job_selectbox"
        )
        selected_job_dict = next(
            (job for job in st.session_state.jobs if f"{job['title']} at {job['company']}" == selected_title),
            None
        )
        st.session_state.selected_job = selected_job_dict
    else:
        st.info("Search for a job to see listings.")


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

    if st.button("Analyze Match with Resume (Llama 3.1)", use_container_width=True, key="analyze_button"):
        if resume_file:
            with st.status("Analyzing resume with Llama 3.1...", expanded=True) as status:
                st.session_state.analysis_runs += 1
                start_time = time.time()
                
                try:
                    resume_text = get_text_from_pdf(resume_file)
                    
                    if not resume_text.strip():
                        st.error("Could not extract text from the uploaded resume. Please try a different PDF.")
                        status.update(label="Analysis failed!", state="error", expanded=True)
                    else:
                        llm_analysis_output = analyze_match_llm(resume_text, job['description'])
                        st.session_state.llm_analysis_results = llm_analysis_output
                        st.session_state.total_runtime += time.time() - start_time
                        status.update(label="Analysis complete!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Analysis failed!", state="error", expanded=True)
                    st.error(f"An error occurred during analysis: {e}")
                    logger.error(f"Error during analysis: {traceback.format_exc()}")
        else:
            st.warning("Please upload your resume as a PDF file first.")

# LLM Analysis Results Section
if st.session_state.llm_analysis_results:
    st.markdown("---")
    st.header("Match Analysis powered by Llama 3.1 🧠")
    st.markdown(st.session_state.llm_analysis_results)
    
    st.markdown("---")
    st.info(f"You've chosen to apply for: **{st.session_state.selected_job['title']}** at **{st.session_state.selected_job['company']}**")
    st.markdown(f"[Apply to this job now]({st.session_state.selected_job['url']})", unsafe_allow_html=True)