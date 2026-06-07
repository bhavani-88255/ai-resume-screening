import streamlit as st
import pandas as pd
import numpy as np
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Salary Data ──
SALARY_DATA = {
    'data scientist': {'base': 900000, 'max': 1800000},
    'machine learning engineer': {'base': 1000000, 'max': 2000000},
    'software engineer': {'base': 700000, 'max': 1600000},
    'web developer': {'base': 500000, 'max': 1200000},
    'data analyst': {'base': 600000, 'max': 1300000},
    'hr manager': {'base': 500000, 'max': 1000000},
    'marketing manager': {'base': 600000, 'max': 1400000},
    'e-commerce specialist': {'base': 500000, 'max': 1100000},
    'devops engineer': {'base': 900000, 'max': 1800000},
    'ui/ux designer': {'base': 500000, 'max': 1200000},
    'mobile developer': {'base': 700000, 'max': 1500000},
    'game developer': {'base': 600000, 'max': 1400000},
    'other': {'base': 400000, 'max': 900000},
}

SKILL_SALARY_BOOST = {
    'machine learning': 80000, 'deep learning': 90000, 'tensorflow': 70000,
    'pytorch': 70000, 'aws': 80000, 'azure': 75000, 'docker': 60000,
    'kubernetes': 70000, 'python': 50000, 'react': 50000, 'sql': 40000,
    'nlp': 85000, 'computer vision': 85000, 'gans': 90000,
    'transformers': 90000, 'reinforcement learning': 90000,
    'flutter': 55000, 'kotlin': 55000, 'swift': 55000,
    'git': 20000, 'agile': 20000, 'leadership': 30000,
}

LOCATION_MULTIPLIER = {
    'bangalore': 1.3, 'mumbai': 1.25, 'delhi': 1.2, 'hyderabad': 1.2,
    'pune': 1.15, 'chennai': 1.15, 'kolkata': 1.0, 'other': 0.9,
}

def extract_text_from_pdf(pdf_file):
    try:
        import pdfplumber
        with pdfplumber.open(pdf_file) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() or ''
        return text
    except:
        return ""

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_match_score(resume_text, job_description):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)
    if not resume_clean or not jd_clean:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(score) * 100, 2)
    except:
        return 0.0

def extract_skills(text):
    skills_list = [
        'python', 'java', 'javascript', 'react', 'html', 'css', 'sql',
        'machine learning', 'deep learning', 'nlp', 'tensorflow', 'pytorch',
        'data analysis', 'excel', 'leadership', 'communication', 'teamwork',
        'seo', 'google ads', 'analytics', 'shopify', 'aws', 'azure',
        'docker', 'kubernetes', 'agile', 'scrum', 'git', 'restful',
        'swift', 'flutter', 'kotlin', 'figma', 'sketch', 'unity',
        'recruitment', 'hr', 'training', 'conflict resolution', 'hris',
        'inventory management', 'customer service', 'marketing',
        'gans', 'transformers', 'reinforcement learning', 'computer vision'
    ]
    text_lower = text.lower()
    return [skill for skill in skills_list if skill in text_lower]

def train_model(df):
    df = df.copy()
    df['resume_clean'] = df['Resume'].apply(clean_text)
    df['jd_clean'] = df['Job_Description'].apply(clean_text)
    df['combined'] = df['resume_clean'] + ' ' + df['jd_clean'] + ' ' + df['Reason_for_decision'].apply(clean_text)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Decision'].str.strip().str.lower())
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['combined'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, vectorizer, le, accuracy

def predict_decision(model, vectorizer, le, resume_text, job_description):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)
    combined = resume_clean + ' ' + jd_clean
    X = vectorizer.transform([combined])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    decision = le.inverse_transform([pred])[0]
    confidence = round(max(proba) * 100, 2)
    return decision, confidence

def send_email(sender_email, sender_password, recipient_email, candidate_name, decision, score, matched_skills, role=""):
    try:
        msg = MIMEMultipart("alternative")
        msg['From'] = sender_email
        msg['To'] = recipient_email
        if decision.lower() == 'select':
            msg['Subject'] = f"Congratulations! You've been Shortlisted — {role}"
            html_body = f"""
            <html><body style="font-family: Arial, sans-serif; padding: 20px;">
            <div style="background-color:#d4edda; padding:20px; border-radius:10px; border-left:5px solid #28a745;">
                <h2 style="color:#28a745;">Congratulations, {candidate_name}!</h2>
                <p>We are pleased to inform you that you have been <b>SHORTLISTED</b> for the role of <b>{role}</b>.</p>
                <hr/>
                <p><b>Match Score:</b> {score}%</p>
                <p><b>Matched Skills:</b> {', '.join([s.title() for s in matched_skills]) if matched_skills else 'N/A'}</p>
                <hr/>
                <p>Our team will contact you soon with the next steps. Best of luck!</p>
                <p style="color:gray; font-size:12px;">This is an automated message from the Resume Screening System.</p>
            </div></body></html>"""
        else:
            msg['Subject'] = f"Application Update — {role}"
            html_body = f"""
            <html><body style="font-family: Arial, sans-serif; padding: 20px;">
            <div style="background-color:#f8d7da; padding:20px; border-radius:10px; border-left:5px solid #dc3545;">
                <h2 style="color:#dc3545;">Application Status Update</h2>
                <p>Dear {candidate_name},</p>
                <p>Thank you for applying for <b>{role}</b>. After careful review, we regret to inform you that your application was <b>not shortlisted</b> at this time.</p>
                <hr/>
                <p><b>Match Score:</b> {score}%</p>
                <p><b>Matched Skills:</b> {', '.join([s.title() for s in matched_skills]) if matched_skills else 'None'}</p>
                <hr/>
                <p>We encourage you to apply for future openings. Thank you for your interest!</p>
                <p style="color:gray; font-size:12px;">This is an automated message from the Resume Screening System.</p>
            </div></body></html>"""
        msg.attach(MIMEText(html_body, "html"))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        return True, "Sent"
    except Exception as e:
        return False, str(e)

def predict_salary(role, experience, location, skills):
    role_lower = role.lower()
    matched_role = 'other'
    for r in SALARY_DATA:
        if r in role_lower:
            matched_role = r
            break
    base = SALARY_DATA[matched_role]['base']
    max_sal = SALARY_DATA[matched_role]['max']
    exp_boost = min(experience * 50000, 500000)
    skill_boost = min(sum(SKILL_SALARY_BOOST.get(s, 0) for s in skills), 400000)
    loc_lower = location.lower()
    loc_mult = LOCATION_MULTIPLIER.get(loc_lower, 0.9)
    min_salary = int((base + exp_boost) * loc_mult)
    max_salary = int((max_sal + exp_boost + skill_boost) * loc_mult)
    avg_salary = int((min_salary + max_salary) / 2)
    return min_salary, max_salary, avg_salary, skill_boost

# ── UI Config ──
st.set_page_config(page_title="Resume Screening System", page_icon="📄", layout="wide")
st.markdown("""
<style>
    .stButton>button { background-color: #4F8BF9; color: white; border-radius: 8px; padding: 0.5em 2em; font-size: 16px; }
    .result-box { padding: 20px; border-radius: 10px; margin: 10px 0; }
    .selected { background-color: #d4edda; border-left: 5px solid #28a745; }
    .rejected { background-color: #f8d7da; border-left: 5px solid #dc3545; }
    .score-box { font-size: 2em; font-weight: bold; text-align: center; padding: 10px; border-radius: 8px; background-color: #e9ecef; }
    .salary-box { padding: 15px; border-radius: 10px; background-color: #fff3cd; border-left: 5px solid #ffc107; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.title("📄 Automated Resume Screening System")
st.markdown("**ML-powered — Resume Screening + Email Notification + Salary Predictor!**")
st.markdown("---")

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Match Score Threshold (%)", 10, 80, 30)
    st.markdown("---")
    st.header("📧 Email Settings")
    sender_email = st.text_input("Your Gmail ID", placeholder="yourmail@gmail.com")
    sender_password = st.text_input("App Password", type="password", placeholder="Gmail App Password")
    if st.button("🔌 Test Connection"):
        if sender_email and sender_password:
            try:
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(sender_email, sender_password)
                server.quit()
                st.success("✅ Connected!")
            except Exception as e:
                st.error(f"❌ Failed: {e}")
        else:
            st.warning("Enter email & password!")
    st.caption("💡 Use Gmail App Password (not your regular Gmail password)")

if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.vectorizer = None
    st.session_state.le = None
    st.session_state.accuracy = None
    st.session_state.df = None

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Train Model", "🔍 Screen Resume", "👥 Bulk Screening",
    "📊 Dataset Analysis", "💰 Salary Predictor"
])

# ── Tab 1 ──
with tab1:
    st.subheader("Upload Dataset & Train Model")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            st.session_state.df = df
            st.success(f"✅ Dataset loaded! {len(df)} records found.")
            st.dataframe(df.head(5), use_container_width=True)
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Records", len(df))
            with col2: st.metric("Selected", df['Decision'].str.lower().str.strip().value_counts().get('select', 0))
            with col3: st.metric("Rejected", df['Decision'].str.lower().str.strip().value_counts().get('reject', 0))
            if st.button("🚀 Train ML Model"):
                with st.spinner("Training model..."):
                    try:
                        model, vectorizer, le, accuracy = train_model(df)
                        st.session_state.model = model
                        st.session_state.vectorizer = vectorizer
                        st.session_state.le = le
                        st.session_state.accuracy = accuracy
                        st.success(f"✅ Model trained! Accuracy: **{accuracy*100:.1f}%**")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Training failed: {e}")
        except Exception as e:
            st.error(f"Error: {e}")
    elif st.session_state.model is not None:
        st.success(f"✅ Model ready! Accuracy: **{st.session_state.accuracy*100:.1f}%**")

# ── Tab 2 ──
with tab2:
    st.subheader("Screen a Single Resume")
    if st.session_state.model is None:
        st.warning("⚠️ Train the model first!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📋 Resume Input")
            input_type = st.radio("Input Type", ["📝 Text", "📄 PDF Upload"], horizontal=True)
            resume_text_final = ""
            if input_type == "📝 Text":
                resume_input = st.text_area("Resume Text", height=250, placeholder="Paste resume here...")
                resume_text_final = resume_input
            else:
                pdf_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
                if pdf_file:
                    with st.spinner("Extracting text..."):
                        extracted = extract_text_from_pdf(pdf_file)
                    if extracted:
                        st.success("✅ PDF extracted!")
                        st.text_area("Extracted Text", extracted, height=200)
                        resume_text_final = extracted
                    else:
                        st.error("❌ Extract failed! Try text input.")
        with col2:
            st.markdown("#### 💼 Job Description")
            jd_input = st.text_area("Job Description", height=180, placeholder="Paste JD here...")
            st.markdown("#### 📧 Candidate Details (for email)")
            candidate_name = st.text_input("Candidate Name", placeholder="John Doe")
            candidate_email = st.text_input("Candidate Email", placeholder="candidate@email.com")
            job_role = st.text_input("Job Role", placeholder="Software Engineer")

        if st.button("🔍 Screen This Resume"):
            if not resume_text_final or not jd_input:
                st.error("Fill both fields!")
            else:
                with st.spinner("Analyzing..."):
                    score = get_match_score(resume_text_final, jd_input)
                    decision, confidence = predict_decision(
                        st.session_state.model, st.session_state.vectorizer,
                        st.session_state.le, resume_text_final, jd_input)
                    matched_skills = list(set(extract_skills(resume_text_final)) & set(extract_skills(jd_input)))
                    missing_skills = list(set(extract_skills(jd_input)) - set(extract_skills(resume_text_final)))

                st.markdown("---")
                st.markdown("### 📊 Results")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"<div class='score-box'>Match Score<br>{score}%</div>", unsafe_allow_html=True)
                with col_b:
                    color = "green" if score >= threshold else "red"
                    label = "✅ Good Match" if score >= threshold else "❌ Low Match"
                    st.markdown(f"<div class='score-box' style='color:{color}'>TF-IDF<br>{label}</div>", unsafe_allow_html=True)
                with col_c:
                    d_color = "green" if decision.lower() == 'select' else "red"
                    d_icon = "✅" if decision.lower() == 'select' else "❌"
                    st.markdown(f"<div class='score-box' style='color:{d_color}'>ML Decision<br>{d_icon} {decision.upper()}<br><small>{confidence}% confident</small></div>", unsafe_allow_html=True)

                st.markdown("---")
                if score >= threshold and decision.lower() == 'select':
                    st.markdown("<div class='result-box selected'>🎉 <b>SHORTLISTED</b> — Strong match!</div>", unsafe_allow_html=True)
                elif score >= threshold:
                    st.markdown("<div class='result-box selected'>⚠️ <b>CONSIDER</b> — Review manually.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='result-box rejected'>❌ <b>REJECTED</b> — Low match.</div>", unsafe_allow_html=True)

                st.markdown("#### 🛠️ Skills Analysis")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**✅ Matched Skills:**")
                    for s in matched_skills: st.markdown(f"- {s.title()}")
                    if not matched_skills: st.write("None detected")
                with c2:
                    st.markdown("**❌ Missing Skills:**")
                    for s in missing_skills: st.markdown(f"- {s.title()}")
                    if not missing_skills: st.write("None detected")

                # Email
                if candidate_email and sender_email and sender_password:
                    st.markdown("---")
                    with st.spinner("Sending email notification..."):
                        ok, msg = send_email(
                            sender_email, sender_password,
                            candidate_email, candidate_name or "Candidate",
                            decision, score, matched_skills, job_role
                        )
                    if ok:
                        st.success(f"📧 Email sent to {candidate_email}!")
                    else:
                        st.error(f"❌ Email failed: {msg}")
                elif candidate_email:
                    st.warning("⚠️ Set Gmail credentials in Sidebar to send email!")

# ── Tab 3 ──
with tab3:
    st.subheader("👥 Bulk Resume Screening")
    st.markdown("**Multiple PDF resumes + One JD → Ranked results + Email notifications!**")
    if st.session_state.model is None:
        st.warning("⚠️ Train the model first!")
    else:
        bulk_role = st.text_input("Job Role", placeholder="Data Scientist", key="bulk_role")
        bulk_jd = st.text_area("Paste JD here", height=150, key="bulk_jd")
        pdf_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)
        st.caption("📧 Enter candidate emails separated by comma (same order as PDFs)")
        bulk_emails_input = st.text_input("Candidate Emails (optional)", placeholder="a@gmail.com, b@gmail.com")
        send_bulk_email = st.checkbox("📨 Send email notifications after screening")

        if pdf_files and bulk_jd:
            if st.button("🚀 Screen All Resumes"):
                bulk_emails = [e.strip() for e in bulk_emails_input.split(',')] if bulk_emails_input else []
                with st.spinner(f"Screening {len(pdf_files)} resumes..."):
                    results = []
                    email_log = []
                    for i, pdf in enumerate(pdf_files):
                        text = extract_text_from_pdf(pdf)
                        if not text:
                            continue
                        score = get_match_score(text, bulk_jd)
                        decision, confidence = predict_decision(
                            st.session_state.model, st.session_state.vectorizer,
                            st.session_state.le, text, bulk_jd)
                        matched = list(set(extract_skills(text)) & set(extract_skills(bulk_jd)))
                        name = pdf.name.replace('.pdf', '')
                        verdict = ('✅ SHORTLISTED' if (score >= threshold and decision.lower() == 'select')
                                   else ('⚠️ CONSIDER' if (score >= threshold)
                                   else '❌ REJECTED'))
                        results.append({
                            'Rank': 0, 'Name': name,
                            'Match Score (%)': score,
                            'ML Decision': decision.upper(),
                            'Confidence (%)': confidence,
                            'Matched Skills': ', '.join(matched) if matched else 'None',
                            'Final Verdict': verdict
                        })
                        if send_bulk_email and sender_email and sender_password and i < len(bulk_emails) and bulk_emails[i]:
                            ok, msg = send_email(sender_email, sender_password, bulk_emails[i], name, decision, score, matched, bulk_role)
                            email_log.append(f"{'✅' if ok else '❌'} {name} → {bulk_emails[i]}")

                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('Match Score (%)', ascending=False)
                results_df['Rank'] = range(1, len(results_df) + 1)

                st.markdown("### 🏆 Ranked Results")
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Total", len(results_df))
                with c2: st.metric("✅ Shortlisted", len(results_df[results_df['Final Verdict'].str.contains('SHORTLISTED', na=False)]))
                with c3: st.metric("❌ Rejected", len(results_df[results_df['Final Verdict'].str.contains('REJECTED', na=False)]))
                st.dataframe(results_df, use_container_width=True)
                st.download_button("📥 Download Results", results_df.to_csv(index=False), "results.csv", "text/csv")
                if email_log:
                    st.markdown("#### 📧 Email Log")
                    for log in email_log:
                        st.write(log)
        else:
            if not bulk_jd: st.warning("⚠️ JD paste pannu!")
            if not pdf_files: st.warning("⚠️ PDF files upload pannu!")

# ── Tab 4 ──
with tab4:
    st.subheader("📊 Dataset Analysis")
    if st.session_state.df is None:
        st.warning("⚠️ Upload dataset first!")
    else:
        df = st.session_state.df
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Decision Distribution")
            st.bar_chart(df['Decision'].str.strip().value_counts())
        with col2:
            st.markdown("#### Top Roles")
            st.bar_chart(df['Role'].value_counts().head(10))
        st.markdown("#### Full Dataset")
        st.dataframe(df[['Role', 'Decision', 'Reason_for_decision']], use_container_width=True)

# ── Tab 5 — SALARY PREDICTOR ──
with tab5:
    st.subheader("💰 Salary Predictor")
    st.markdown("**Enter your details to get estimated salary range in India!**")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        sal_role = st.selectbox("🎯 Job Role", [
            'Data Scientist', 'Machine Learning Engineer', 'Software Engineer',
            'Web Developer', 'Data Analyst', 'HR Manager', 'Marketing Manager',
            'E-commerce Specialist', 'DevOps Engineer', 'UI/UX Designer',
            'Mobile Developer', 'Game Developer', 'Other'
        ])
        sal_exp = st.slider("📅 Years of Experience", 0, 20, 2)
        sal_location = st.selectbox("📍 Location", [
            'Bangalore', 'Mumbai', 'Delhi', 'Hyderabad',
            'Pune', 'Chennai', 'Kolkata', 'Other'
        ])
    with col2:
        st.markdown("#### 🛠️ Your Skills")
        all_skills = [
            'python', 'java', 'javascript', 'react', 'sql', 'machine learning',
            'deep learning', 'nlp', 'tensorflow', 'pytorch', 'aws', 'azure',
            'docker', 'kubernetes', 'git', 'flutter', 'kotlin', 'swift',
            'computer vision', 'gans', 'transformers', 'reinforcement learning',
            'figma', 'agile', 'leadership', 'communication'
        ]
        selected_skills = st.multiselect("Select your skills", all_skills)
        sal_resume = st.text_area("Or paste Resume (auto-detect skills)", height=130, placeholder="Paste resume here...")

    if st.button("💰 Predict My Salary"):
        auto_skills = extract_skills(sal_resume) if sal_resume else []
        combined_skills = list(set(selected_skills + auto_skills))
        min_sal, max_sal, avg_sal, skill_boost = predict_salary(sal_role, sal_exp, sal_location, combined_skills)

        st.markdown("---")
        st.markdown("### 💰 Estimated Salary Range")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='score-box'>Minimum<br>₹{min_sal:,}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='score-box' style='color:green;'>Average<br>₹{avg_sal:,}</div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='score-box'>Maximum<br>₹{max_sal:,}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
        <div class='salary-box'>
        <h4>📊 Breakdown</h4>
        <p>🎯 <b>Role:</b> {sal_role} | 📍 <b>Location:</b> {sal_location}</p>
        <p>📅 <b>Experience Boost:</b> ₹{min(sal_exp * 50000, 500000):,}</p>
        <p>🛠️ <b>Skills Boost:</b> ₹{min(skill_boost, 400000):,}</p>
        <p>✅ <b>Detected Skills:</b> {', '.join([s.title() for s in combined_skills]) if combined_skills else 'None'}</p>
        </div>
        """, unsafe_allow_html=True)

        chart_data = pd.DataFrame({
            'Type': ['Minimum', 'Average', 'Maximum'],
            'Salary (INR)': [min_sal, avg_sal, max_sal]
        }).set_index('Type')
        st.bar_chart(chart_data)

        missing_boost_skills = [s for s in SKILL_SALARY_BOOST if s not in combined_skills][:3]
        if missing_boost_skills:
            st.info(f"💡 **Tip:** Learn {', '.join([s.title() for s in missing_boost_skills])} to boost your salary further!")
