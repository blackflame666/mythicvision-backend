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

# --- ULTIMATE COACH AI SCREENSHOT ANALYSIS ---
@app.post("/api/gameplay/analyze-screenshot")
async def analyze_screenshot(
    file: UploadFile = File(...),
    player_name: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a post-game screenshot and get a massive, detailed AI coaching report"""
    
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Only image files (JPG/PNG) are allowed.")
    
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024: 
        raise HTTPException(status_code=400, detail="Image size must be less than 5MB")
    
    base64_image = base64.b64encode(contents).decode('utf-8')

    # THE ULTIMATE COACH PROMPT
    system_prompt = """You are the world's greatest Mobile Legends: Bang Bang Esports Coach. You are known for your brutal honesty, deep game knowledge, and ability to find hidden mistakes in player data. 
    You do not just read stats; you deduce gameplay behavior from them.
    You must output ONLY valid JSON. No markdown formatting, no code blocks, just raw JSON."""

    user_prompt = f"""Analyze this Mobile Legends post-game scoreboard for the player named: "{player_name}"

    YOUR ANALYSIS FRAMEWORK (Be extremely detailed and write long paragraphs):
    
    1. IDENTIFY THE PLAYER: Find the exact row for "{player_name}". Note their Hero, KDA, Gold, Items, and Score.
    2. DEDUCE ROLE & EXPECTATIONS: Based on the hero, what was their job? (e.g., If Tank, they should peel and initiate. If Marksman, they should farm and deal damage).
    3. ANALYZE KDA CONTEXT: 
       - High Kills/Low Deaths: Good mechanics, but did they secure objectives?
       - Low Kills/High Deaths: Positioning errors, getting caught out, or diving too deep.
       - High Assists: Good team player, but did they rely too much on teammates?
    4. ITEMIZATION FORENSICS: Look at their 6 items. 
       - Did they build the right items for the enemy team composition? (e.g., Building physical defense against a magic damage team is a fatal mistake).
       - Did they finish their core build?
    5. MACRO & ECONOMY: Look at their Gold compared to teammates. 
       - If they have low gold but high kills, they are killing but not farming (bad macro).
       - If they have high gold but low impact, they are farming but not fighting (cowardly play).
    6. MISTAKE AUTOPSY: Based on the above, deduce their biggest mistake. (e.g., "You died 8 times because you were building squishy damage items as a Tank, or you were overextending without vision.")

    Generate a MASSIVE, comprehensive coaching report. Do not be brief. Write like a professional analyst.
    
    Output strictly in this JSON format:
    {json.dumps({
        "match_summary": "string (e.g., Victory | 12:14 Duration | 2/4/20 KDA)",
        "overall_rating": "string (e.g., 7.5/10 - Good mechanics but poor macro)",
        "analysis": {
            "gameplay_mechanics": {
                "title": "Gameplay & Mechanics Deep Dive",
                "overall_score": "string",
                "detailed_analysis": "string (Write 3-4 long paragraphs analyzing their KDA, score, and role performance. Deduce their mechanical skill level.)",
                "strengths": ["string (detailed point)", "string (detailed point)", "string (detailed point)"],
                "weaknesses": ["string (detailed point)", "string (detailed point)", "string (detailed point)"],
                "actionable_tips": ["string (specific drill or habit)", "string (specific drill or habit)", "string (specific drill or habit)"]
            },
            "mistakes_corrections": {
                "title": "Critical Mistakes & The 'Why'",
                "critical_errors": [
                    {
                        "mistake": "string (e.g., Overextending without vision)",
                        "evidence": "string (e.g., You died 8 times but only had 10k gold, meaning you died early and missed farm)",
                        "correction": "string (Exactly what they should have done instead)"
                    },
                    {
                        "mistake": "string",
                        "evidence": "string",
                        "correction": "string"
                    }
                ]
            },
            "itemization_macro": {
                "title": "Itemization & Macro Strategy",
                "overall_score": "string",
                "items_built": ["string (list the 6 items you see)"],
                "detailed_analysis": "string (Write 2-3 paragraphs analyzing if their items were optimal. Did they counter the enemy? Did they build greedily?)",
                "macro_assessment": "string (Analyze their gold and game duration. Did they play for early game or late game?)",
                "actionable_tips": ["string (specific item advice)", "string (specific macro advice)"]
            }
        },
        "overall_recommendations": {
            "priority_1": "string (The #1 thing they must fix to rank up)",
            "priority_2": "string",
            "priority_3": "string",
            "long_term_goal": "string (What rank can they reach if they fix these issues?)"
        }
    }, indent=2)}"""

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
            max_tokens=3000 # Increased for longer output
        )
        
        content = response.choices[0].message.content
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        
        analysis_data = json.loads(content)
        analysis_data["file_id"] = f"screenshot_{current_user.id}_{datetime.utcnow().timestamp()}"
        analysis_data["player_focus"] = player_name
        
        return analysis_data

    except json.JSONDecodeError:
        print("Raw AI Output:", content)
        raise HTTPException(status_code=500, detail="AI returned invalid JSON. Please try a clearer screenshot.")
    except Exception as e:
        print(f"OpenAI Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI Analysis failed: {str(e)}")

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
