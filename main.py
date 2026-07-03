import os
import json
import re
import base64
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from jose import jwt, JWTError
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

# Load environment variables
load_dotenv()

# --- APP INITIALIZATION ---
app = FastAPI(title="MythicVision Backend - ML Coach")

# --- OPENAI SETUP ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set. AI analysis will fail.")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- DATABASE SETUP ---
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mlcoach.db")
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String)
    avatar_url = Column(String, nullable=True)
    google_id = Column(String, unique=True)
    is_premium = Column(Boolean, default=False)
    plan_type = Column(String, default="free")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Base.metadata.create_all(bind=engine)

with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE users ADD COLUMN plan_type VARCHAR DEFAULT 'free'"))
        conn.commit()
    except Exception:
        conn.rollback()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- CONFIGURATION ---
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://mlcoach.online")
API_URL = os.getenv("API_URL", "https://mythicvision-backend.onrender.com")
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-change-this-in-production")
ALGORITHM = "HS256"

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, https_only=False, same_site="lax", max_age=3600)

oauth = OAuth()
oauth.register(name='google', client_id=os.getenv("GOOGLE_CLIENT_ID"), client_secret=os.getenv("GOOGLE_CLIENT_SECRET"), server_metadata_url='https://accounts.google.com/.well-known/openid-configuration', client_kwargs={'scope': 'openid email profile'})

class AnalyzeRequest(BaseModel):
    hero_name: str = None
class UpgradeRequest(BaseModel):
    plan_name: str

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=7))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.headers.get("Authorization")
    if not token: raise HTTPException(status_code=401, detail="Not authenticated")
    if token.startswith("Bearer "): token = token.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None: raise HTTPException(status_code=401, detail="Invalid token")
        user = db.query(User).filter(User.email == email).first()
        if user is None: raise HTTPException(status_code=404, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- ROUTES ---
@app.get("/")
def root():
    return {"message": "MythicVision API is running!", "docs": "/docs"}

@app.get("/auth/google/login")
async def google_login(request: Request):
    return await oauth.google.authorize_redirect(request, f"{API_URL}/auth/google/callback")

@app.get("/auth/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
        if not token: raise HTTPException(status_code=400, detail="Failed to get access token")
        user_info = token.get('userinfo')
        if not user_info:
            resp = await oauth.google.get('https://www.googleapis.com/oauth2/v3/userinfo', token=token)
            user_info = resp.json()
        
        email, name, avatar, google_id = user_info.get('email'), user_info.get('name'), user_info.get('picture'), user_info.get('sub')
        db_user = db.query(User).filter(User.email == email).first()
        if not db_user:
            db_user = User(email=email, name=name, avatar_url=avatar, google_id=google_id, is_premium=False, plan_type="free")
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
        else:
            db_user.last_login = datetime.utcnow()
            db.commit()
        
        access_token = create_access_token(data={"sub": email, "user_id": db_user.id, "name": db_user.name})
        return RedirectResponse(url=f"{FRONTEND_URL}/dashboard?token={access_token}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

@app.get("/api/me")
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "email": current_user.email, "name": current_user.name, "avatar_url": current_user.avatar_url, "is_premium": current_user.is_premium or False, "plan_type": current_user.plan_type or "free", "created_at": current_user.created_at, "last_login": current_user.last_login}

# --- REAL AI SCREENSHOT ANALYSIS ENDPOINT ---
@app.post("/api/gameplay/analyze-screenshot")
async def analyze_screenshot(
    file: UploadFile = File(...),
    hero_name: str = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Upload a post-game screenshot and get REAL AI analysis"""
    
    # 1. Validate File
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Only image files (JPG/PNG) are allowed for analysis.")
    
    # 2. Check Elite status if hero_name is provided
    if hero_name and hero_name.lower() != "null" and current_user.plan_type != "elite":
        raise HTTPException(status_code=403, detail="Hero-specific analysis is an Elite-exclusive feature.")

    # 3. Read and Encode Image
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024: # 5MB limit for images
        raise HTTPException(status_code=400, detail="Image size must be less than 5MB")
    
    base64_image = base64.b64encode(contents).decode('utf-8')

    # 4. Prepare AI Prompt
    hero_context = f"The user states they played the hero: {hero_name}. " if (hero_name and hero_name.lower() != "null") else ""
    
    system_prompt = """You are a professional Mobile Legends: Bang Bang esports coach. You analyze post-game screenshots to provide detailed, actionable coaching. 
    You must output ONLY valid JSON. No markdown formatting, no code blocks, just raw JSON.
    Analyze the visible stats (KDA, Items, Gold, Damage, etc.) and generate a report."""

    user_prompt = f"""{hero_context}Analyze this MLBB post-game screenshot. Extract the stats. Generate a detailed coaching report based on these stats. 
    Output strictly in this JSON format: 
    {
        "match_summary": "string (e.g., Victory | 8/3/12 KDA)",
        "overall_rating": "string (e.g., 8.5/10)",
        "analysis": {
            "gameplay_mechanics": {
                "title": "Gameplay & Mechanics Analysis",
                "overall_score": "string",
                "detailed_analysis": "string (2-3 paragraphs analyzing performance based on KDA and damage)",
                "strengths": ["string", "string"],
                "weaknesses": ["string", "string"],
                "actionable_tips": ["string", "string"]
            },
            "mistakes_corrections": {
                "title": "Critical Mistakes & Detailed Corrections",
                "critical_errors": [
                    {
                        "time": "string (e.g., Late Game)",
                        "severity": "string",
                        "what_happened": "string (infer from stats, e.g., high deaths)",
                        "why_it_was_wrong": "string",
                        "how_to_fix": "string"
                    }
                ]
            },
            "positioning_rotations": {
                "title": "Positioning & Rotations Deep Dive",
                "overall_score": "string",
                "detailed_analysis": "string",
                "actionable_tips": ["string", "string"]
            },
            "itemization_macro": {
                "title": "Itemization & Macro Strategy",
                "overall_score": "string",
                "items_built": ["string (list items seen in screenshot)"],
                "detailed_analysis": "string (analyze if items are good for the enemy comp)",
                "actionable_tips": ["string", "string"]
            }
        },
        "overall_recommendations": {
            "priority_1": "string",
            "priority_2": "string",
            "priority_3": "string"
        }
    }"""

    # 5. Call OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=2500
        )
        
        # 6. Parse JSON Response
        content = response.choices[0].message.content
        # Remove markdown code blocks if the AI adds them
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        
        analysis_data = json.loads(content)
        
        # Add metadata
        analysis_data["file_id"] = f"screenshot_{current_user.id}_{datetime.utcnow().timestamp()}"
        analysis_data["hero_focus"] = hero_name if (hero_name and hero_name.lower() != "null") else "overall gameplay"
        
        return analysis_data

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON. Please try a clearer screenshot.")
    except Exception as e:
        print(f"OpenAI Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI Analysis failed: {str(e)}")

# --- LEGACY VIDEO UPLOAD (Kept for now, but screenshot is preferred) ---
@app.post("/api/gameplay/upload")
async def upload_gameplay(file: UploadFile = File(...), hero_name: str = Form(None), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # (Keep existing video upload code if needed, or remove it)
    return {"message": "Please use the screenshot upload endpoint for real AI analysis."}

@app.get("/api/gameplay/{file_id}")
async def get_analysis_results(file_id: str, current_user: User = Depends(get_current_user)):
    # Since we are doing real-time analysis now, this endpoint might need to fetch from a DB.
    # For now, we will rely on the frontend storing the result from the analyze-screenshot response.
    raise HTTPException(status_code=404, detail="Use the analyze-screenshot endpoint for real-time results.")

@app.post("/api/subscription/upgrade")
async def upgrade_subscription(request: UpgradeRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    plan = request.plan_name.lower()
    if plan not in ["pro", "elite"]: raise HTTPException(status_code=400, detail="Invalid plan name.")
    current_user.is_premium = True
    current_user.plan_type = plan
    db.commit()
    return {"message": f"Successfully upgraded to {plan}!", "is_premium": True, "plan_type": plan}

@app.post("/api/user/avatar")
async def upload_avatar(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types: raise HTTPException(status_code=400, detail="Only JPG and PNG images allowed")
    contents = await file.read()
    if len(contents) > 2 * 1024 * 1024: raise HTTPException(status_code=400, detail="File size must be less than 2MB")
    upload_dir = Path("uploads/avatars")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{current_user.id}_{file.filename}"
    with open(file_path, "wb") as buffer: buffer.write(contents)
    avatar_url = f"/uploads/avatars/{current_user.id}_{file.filename}"
    current_user.avatar_url = avatar_url
    db.commit()
    return {"avatar_url": avatar_url, "message": "Avatar updated successfully"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "mythicvision-backend"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
