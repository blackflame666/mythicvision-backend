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

load_dotenv()

app = FastAPI(title="MythicVision Backend - ML Coach")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set.")
client = OpenAI(api_key=OPENAI_API_KEY)

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

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://mlcoach.online")
API_URL = os.getenv("API_URL", "https://mythicvision-backend.onrender.com")
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-change-this-in-production")
ALGORITHM = "HS256"

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, https_only=False, same_site="lax", max_age=3600)

oauth = OAuth()
oauth.register(name='google', client_id=os.getenv("GOOGLE_CLIENT_ID"), client_secret=os.getenv("GOOGLE_CLIENT_SECRET"), server_metadata_url='https://accounts.google.com/.well-known/openid-configuration', client_kwargs={'scope': 'openid email profile'})

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
    return {
        "id": current_user.id, "email": current_user.email, "name": current_user.name, 
        "avatar_url": current_user.avatar_url, "is_premium": current_user.is_premium or False, 
        "plan_type": current_user.plan_type or "free", "created_at": current_user.created_at, 
        "last_login": current_user.last_login
    }

@app.post("/api/gameplay/analyze-screenshot")
async def analyze_screenshot(
    file: UploadFile = File(...),
    player_name: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Only image files (JPG/PNG) are allowed.")
    
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024: 
        raise HTTPException(status_code=400, detail="Image size must be less than 5MB")
    
    base64_image = base64.b64encode(contents).decode('utf-8')

    system_prompt = """You are the world's greatest Mobile Legends: Bang Bang Esports Coach.
    RULES:
    1. NEVER guess. If you can't see it, infer it from other stats. NEVER use "N/A" or "Data not available".
    2. HERO IDENTIFICATION: Only identify the hero if the portrait next to the username is clearly visible. If you are unsure, DO NOT name the hero; refer to them as "The player".
    3. ERROR HANDLING: If you cannot find the player name "{player_name}" anywhere on the scoreboard, you must return ONLY this JSON: { "error": "Player not found", "message": "The name '{player_name}' does not appear on this scoreboard." }"""

    user_prompt = f"""Analyze this scoreboard for: "{player_name}".
    
    1. Find "{player_name}" by reading the text usernames.
    2. Extract: KDA, Gold, Items (all 6), Score/Rating, Win/Loss, Duration.
    3. If the hero portrait is clear, identify it. If not, just say "Played a [Role] hero".
    4. Provide a MASSIVE, detailed analysis. NO "N/A".
    
    Return JSON:
    {{
        "match_summary": "Victory/Defeat | Duration: XX:XX | KDA: X/X/X",
        "overall_rating": "X.X/10 - Summary",
        "duration": "XX:XX",
        "result": "Victory/Defeat",
        "hero_played": "Hero Name OR 'Unknown Hero (Role)'",
        "analysis": {{
            "gameplay_mechanics": {{
                "title": "Gameplay & Mechanics Deep Dive",
                "overall_score": "XX/100",
                "detailed_analysis": "4-5 LONG paragraphs analyzing KDA, Score, and Role performance. Cite specific numbers.",
                "strengths": ["Strength 1 (2-3 sentences)", "Strength 2", "Strength 3", "Strength 4"],
                "weaknesses": ["Weakness 1", "Weakness 2", "Weakness 3", "Weakness 4"],
                "actionable_tips": ["Tip 1", "Tip 2", "Tip 3", "Tip 4", "Tip 5"]
            }},
            "mistakes_corrections": {{
                "title": "Critical Mistakes & The Why",
                "critical_errors": [
                    {{"mistake": "Mistake 1", "evidence": "Evidence from stats", "correction": "Correction"}},
                    {{"mistake": "Mistake 2", "evidence": "Evidence", "correction": "Correction"}},
                    {{"mistake": "Mistake 3", "evidence": "Evidence", "correction": "Correction"}},
                    {{"mistake": "Mistake 4", "evidence": "Evidence", "correction": "Correction"}}
                ]
            }},
            "itemization_macro": {{
                "title": "Itemization & Macro Strategy",
                "overall_score": "XX/100",
                "items_built": ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6"],
                "detailed_analysis": "4-5 LONG paragraphs analyzing items vs enemy team. Cite specific items.",
                "macro_assessment": "3-4 paragraphs analyzing Gold efficiency and game duration impact.",
                "actionable_tips": ["Tip 1", "Tip 2", "Tip 3", "Tip 4", "Tip 5"]
            }}
        }},
        "overall_recommendations": {{
            "priority_1": "Priority 1 (3-4 sentences)",
            "priority_2": "Priority 2 (3-4 sentences)",
            "priority_3": "Priority 3 (3-4 sentences)",
            "long_term_goal": "Long term goal (3-4 sentences)"
        }}
    }}"""

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
            max_tokens=4000
        )
        
        content = response.choices[0].message.content
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        
        analysis_data = json.loads(content)
        
        # CHECK FOR ERROR RESPONSE (Player not found)
        if "error" in analysis_data and analysis_data["error"] == "Player not found":
            raise HTTPException(status_code=404, detail=analysis_data["message"])

        analysis_data["file_id"] = f"screenshot_{current_user.id}_{datetime.utcnow().timestamp()}"
        analysis_data["player_focus"] = player_name
        
        return analysis_data

    except json.JSONDecodeError as e:
        print("JSON Error:", content)
        raise HTTPException(status_code=500, detail="AI returned invalid JSON.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed.")

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
