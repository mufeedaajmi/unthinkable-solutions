import os, json, re, sqlite3, tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
app = FastAPI(title='Smart Resume Screener', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
DB_FILE = 'resume_screener.db'
SKILLS = ['Python','Java','C++','C','JavaScript','TypeScript','HTML','CSS','React','Node.js','Express','FastAPI','Flask','Django','Spring','SQL','MySQL','PostgreSQL','MongoDB','AWS','Azure','Docker','Kubernetes','Git','GitHub','Machine Learning','Deep Learning','NLP','TensorFlow','PyTorch','Pandas','NumPy','Scikit-learn','REST API','Data Structures','Algorithms','Power BI','Tableau','Excel','Data Analysis']


def db():
    c = sqlite3.connect(DB_FILE)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    c = db()
    c.execute('''CREATE TABLE IF NOT EXISTS resumes (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, name TEXT, email TEXT, phone TEXT, skills TEXT, education TEXT, experience TEXT, resume_text TEXT, match_score REAL, result TEXT)''')
    c.commit(); c.close()
init_db()


def extract_text(path):
    ext = Path(path).suffix.lower()
    if ext == '.pdf':
        r = PdfReader(path)
        return '\n'.join(p.extract_text() or '' for p in r.pages)
    if ext == '.docx':
        d = Document(path)
        return '\n'.join(p.text for p in d.paragraphs)
    if ext == '.txt':
        return Path(path).read_text(encoding='utf-8', errors='ignore')
    raise ValueError('Only PDF, DOCX and TXT files are supported.')


def extract_data(text):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    email_m = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    phone_m = re.search(r'(?:\+91[\s-]?)?[6-9]\d{9}', text)
    name = 'Unknown Candidate'
    for line in lines[:10]:
        if '@' not in line and not any(ch.isdigit() for ch in line) and 1 < len(line.split()) <= 5 and len(line) <= 60:
            name = line; break
    skills = [s for s in SKILLS if s.lower() in text.lower()]
    edu_keys = ['b.tech','btech','b.e','bachelor','m.tech','mtech','master','mca','mba','phd','university','college']
    exp_keys = ['intern','internship','experience','developer','engineer','analyst','software','project']
    education = [x for x in lines if any(k in x.lower() for k in edu_keys)][:10]
    experience = [x for x in lines if any(k in x.lower() for k in exp_keys)][:20]
    return {'name': name, 'email': email_m.group(0) if email_m else '', 'phone': phone_m.group(0) if phone_m else '', 'skills': skills, 'education': education, 'experience': experience}

SYSTEM_PROMPT = '''You are an expert technical recruiter. Compare a resume with a job description using only job-relevant qualifications: skills, relevant experience, education, and role fit. Do not use age, gender, race, religion, nationality, marital status, disability, photograph, appearance, or other protected/unrelated characteristics. Do not claim a skill unless the resume provides evidence. Return ONLY valid JSON with: overall_score, skill_score, experience_score, education_score, role_fit_score, matched_skills, missing_skills, strengths, weaknesses, recommendation, justification. Scores are 1-10. Recommendation must be SHORTLIST, MAYBE, or REJECT. Use 8-10 SHORTLIST, 6-7.9 MAYBE, 1-5.9 REJECT.'''


def llm_screen(data, resume_text, job):
    key = os.getenv('OPENAI_API_KEY')
    if not key: raise ValueError('OPENAI_API_KEY is not configured. Add it to .env and restart the backend.')
    client = OpenAI(api_key=key)
    prompt = f'''JOB DESCRIPTION:\n{job}\n\nCANDIDATE RESUME:\n{resume_text}\n\nEXTRACTED DATA:\n{json.dumps(data, indent=2)}\n\nCompare the candidate with the job description and return only the requested JSON.'''
    r = client.chat.completions.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'), temperature=0.1, response_format={'type':'json_object'}, messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':prompt}])
    result = json.loads(r.choices[0].message.content)
    for k in ['overall_score','skill_score','experience_score','education_score','role_fit_score']:
        result[k] = max(1, min(10, float(result.get(k,1))))
    result['recommendation'] = str(result.get('recommendation','MAYBE')).upper()
    if result['recommendation'] not in {'SHORTLIST','MAYBE','REJECT'}: result['recommendation']='MAYBE'
    return result

@app.get('/')
def root(): return {'message':'Smart Resume Screener backend is running','docs':'http://127.0.0.1:8000/docs'}
@app.get('/health')
def health(): return {'status':'healthy'}

@app.post('/upload-resume')
async def upload_resume(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in {'.pdf','.docx','.txt'}: raise HTTPException(400,'Only PDF, DOCX and TXT files are supported.')
    temp_path = None
    try:
        content = await file.read()
        if not content: raise HTTPException(400,'The uploaded file is empty.')
        with tempfile.NamedTemporaryFile(delete=False,suffix=ext) as f:
            f.write(content); temp_path=f.name
        text = extract_text(temp_path).strip()
        if not text: raise HTTPException(400,'No readable text was found in the resume.')
        data = extract_data(text)
        c=db(); cur=c.execute('INSERT INTO resumes (filename,name,email,phone,skills,education,experience,resume_text) VALUES (?,?,?,?,?,?,?,?)',(file.filename,data['name'],data['email'],data['phone'],json.dumps(data['skills']),json.dumps(data['education']),json.dumps(data['experience']),text)); c.commit(); rid=cur.lastrowid; c.close()
        return {'message':'Resume uploaded successfully','resume_id':rid,**data}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500,str(e))
    finally:
        if temp_path and os.path.exists(temp_path): os.remove(temp_path)

@app.post('/screen')
async def screen_resume(resume_id:int=Form(...), job_description:str=Form(...)):
    if not job_description.strip(): raise HTTPException(400,'Job description cannot be empty.')
    c=db(); row=c.execute('SELECT * FROM resumes WHERE id=?',(resume_id,)).fetchone()
    if not row: c.close(); raise HTTPException(404,'Resume not found.')
    data={'name':row['name'],'email':row['email'],'phone':row['phone'],'skills':json.loads(row['skills'] or '[]'),'education':json.loads(row['education'] or '[]'),'experience':json.loads(row['experience'] or '[]')}
    try:
        result=llm_screen(data,row['resume_text'],job_description)
        c.execute('UPDATE resumes SET match_score=?, result=? WHERE id=?',(result['overall_score'],json.dumps(result),resume_id)); c.commit()
        return {'resume_id':resume_id,'candidate':row['name'],'screening':result}
    except Exception as e: raise HTTPException(500,str(e))
    finally: c.close()

@app.get('/candidates')
def candidates():
    c=db(); rows=c.execute('SELECT id,filename,name,email,skills,match_score,result FROM resumes ORDER BY CASE WHEN match_score IS NULL THEN 1 ELSE 0 END, match_score DESC').fetchall(); c.close()
    return [{'id':r['id'],'filename':r['filename'],'name':r['name'],'email':r['email'],'skills':json.loads(r['skills'] or '[]'),'score':r['match_score'],'screening':json.loads(r['result']) if r['result'] else None} for r in rows]
