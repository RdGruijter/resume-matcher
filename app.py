import streamlit as st
import spacy
import re

# This code assumes the 'en_core_web_md' model folder is in the same directory.
@st.cache_resource
def load_spacy_model():
    """
    Loads the SpaCy model with caching.
    """
    try:
        # The model is loaded directly from the folder in the repository.
        return spacy.load("./en_core_web_md")
    except OSError:
        st.error("SpaCy model 'en_core_web_md' not found. Please ensure the model folder is in the same directory.")
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
    return filtered_keywords


def calculate_semantic_score(jd_keywords, resume_keywords, similarity_threshold=0.7):
    """
    Calculates a match score based on both exact keyword matches and semantic similarity.
    """
    if not jd_keywords:
        return 100, []

    matched_keywords = jd_keywords.intersection(resume_keywords)
    
    semantically_matched = set()
    for jd_kw in jd_keywords:
        if jd_kw not in matched_keywords:
            jd_token = nlp(jd_kw)
            if jd_token.has_vector:
                for res_kw in resume_keywords:
                    res_token = nlp(res_kw)
                    if res_token.has_vector and jd_token.similarity(res_token) > similarity_threshold:
                        semantically_matched.add(jd_kw)
                        break
                            
    all_matched = matched_keywords.union(semantically_matched)
    
    score = (len(all_matched) / len(jd_keywords)) * 100
    truly_missing = jd_keywords - all_matched
    
    return round(score, 2), truly_missing


def parse_resume_sections(resume_text):
    """
    Parses resume text into sections based on common headers.
    """
    sections = {}
    pattern = re.compile(r'^(?:summary|profile|experience|work history|education|skills|projects|certifications)\s*$', re.MULTILINE | re.IGNORECASE)
    
    matches = list(pattern.finditer(resume_text))
    
    if not matches:
        return {"Uncategorized": resume_text}
        
    for i, match in enumerate(matches):
        section_name = match.group(0).strip().lower()
        start_index = match.end()
        end_index = matches[i+1].start() if i+1 < len(matches) else len(resume_text)
        
        section_content = resume_text[start_index:end_index].strip()
        sections[section_name] = section_content
        
    return sections


def generate_contextual_suggestions(missing_keywords, sections):
    """
    Generates specific, section-based suggestions for truly missing keywords.
    """
    suggestions = {}
    
    skill_keywords = {"python", "aws", "machine learning", "pytorch", "gcp", "azure", "sql", "tableau", "docker", "r", "predictive modeling", "statistical modeling", "data analysis", "data analytics", "data visualization", "a/b testing"}
    experience_verbs = {"develop", "manage", "led", "implement", "create", "design", "build", "conduct", "deploy", "leverage"}
    
    for keyword in missing_keywords:
        if any(kw in keyword.lower() for kw in skill_keywords):
            target_section = "skills"
            if target_section in sections:
                suggestions.setdefault(target_section, []).append(f"💡 Consider adding '{keyword.title()}' to your **{target_section.capitalize()}** section.")
            else:
                suggestions.setdefault("General", []).append(f"💡 The job requires '{keyword.title()}'. You may want to add a dedicated **{target_section.capitalize()}** section to your resume.")
                
        elif any(verb in keyword.lower() for verb in experience_verbs):
            target_section = "experience"
            if target_section in sections:
                suggestions.setdefault(target_section, []).append(f"💡 Add a bullet point in your **{target_section.capitalize()}** section that uses the term '{keyword.title()}', for example: '...'{keyword.title()}ed a scalable machine learning model...'")
            else:
                suggestions.setdefault("General", []).append(f"💡 The job description mentions '{keyword.title()}'. Consider adding an **{target_section.capitalize()}** section to showcase relevant projects or roles.")
        
        else:
            suggestions.setdefault("General", []).append(f"💡 The job description mentions '{keyword.title()}'. Consider adding it to a relevant part of your resume.")
                
    return suggestions


# --- Streamlit UI ---
st.set_page_config(page_title="AI Job Assistant PoC: Resume Matcher", layout="wide")
st.title("AI Job Assistant: Resume & Job Description Matcher (PoC)")
st.write("Enter your resume and a job description below to see how well they align.")

col1, col2 = st.columns(2)

with col1:
    st.header("Your Resume")
    resume_input = st.text_area("Paste your resume text here:", height=300,
                                placeholder="E.g., Experienced Software Engineer with strong Python and ML skills...")

with col2:
    st.header("Job Description")
    jd_input = st.text_area("Paste the job description text here:", height=300,
                            placeholder="E.g., We are looking for a Machine Learning Engineer to design and develop AI applications...")

if st.button("Analyze Resume vs. Job Description"):
    if resume_input and jd_input:
        with st.spinner("Analyzing..."):
            jd_keywords = extract_keywords_spacy(jd_input)
            resume_keywords = extract_keywords_spacy(resume_input)
            
            match_score, truly_missing_keywords = calculate_semantic_score(jd_keywords, resume_keywords)

            if 'results' not in st.session_state:
                st.session_state.results = {}
            st.session_state.results = {
                "match_score": match_score,
                "missing_keywords": truly_missing_keywords,
                "jd_keywords": jd_keywords,
                "resume_keywords": resume_keywords,
                "initial_resume_text": resume_input
            }
    else:
        st.warning("Please paste text into both the Resume and Job Description fields.")

if 'results' in st.session_state and st.session_state.results:
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
