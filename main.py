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
    plan_type = Column(String, default="free")  # "free", "pro", or "elite"
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Add plan_type column to existing users (database migration)
with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE users ADD COLUMN plan_type VARCHAR DEFAULT 'free'"))
        conn.commit()
        print("Added plan_type column to users table")
    except Exception as e:
        print(f"plan_type column already exists or migration completed: {e}")
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

# 1. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Session Middleware (CRITICAL for OAuth state/CSRF protection)
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=False,
    same_site="lax",
    max_age=3600
)

# --- GOOGLE OAUTH SETUP ---
oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile',
    },
)

# --- PYDANTIC MODELS ---
class UpgradeRequest(BaseModel):
    plan_name: str

# --- HELPER FUNCTIONS ---
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=7))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if token.startswith("Bearer "):
        token = token.split(" ")[1]

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- ROUTES ---

@app.get("/")
def root():
    return {"message": "MythicVision API is running!", "docs": "/docs"}

@app.get("/auth/google/login")
async def google_login(request: Request):
    """Step 1: Redirect user to Google for authentication"""
    redirect_uri = f"{API_URL}/auth/google/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    """Step 2: Google redirects back here with user info"""
    try:
        token = await oauth.google.authorize_access_token(request)
        if not token:
            raise HTTPException(status_code=400, detail="Failed to get access token")
        
        user_info = token.get('userinfo')
        if not user_info:
            resp = await oauth.google.get('https://www.googleapis.com/oauth2/v3/userinfo', token=token)
            user_info = resp.json()
        
        email = user_info.get('email')
        name = user_info.get('name')
        avatar = user_info.get('picture')
        google_id = user_info.get('sub')

        # Check if user exists, create if new
        db_user = db.query(User).filter(User.email == email).first()
        if not db_user:
            db_user = User(
                email=email, 
                name=name, 
                avatar_url=avatar, 
                google_id=google_id, 
                is_premium=False,
                plan_type="free"
            )
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
        else:
            db_user.last_login = datetime.utcnow()
            db.commit()

        # Generate JWT Token
        access_token = create_access_token(data={"sub": email, "user_id": db_user.id, "name": db_user.name})
        return RedirectResponse(url=f"{FRONTEND_URL}/dashboard?token={access_token}")
    except Exception as e:
        print(f"Auth Error: {e}")
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

@app.get("/api/me")
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get the currently logged-in user's profile"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "avatar_url": current_user.avatar_url,
        "is_premium": current_user.is_premium or False,
        "plan_type": current_user.plan_type or "free",
        "created_at": current_user.created_at,
        "last_login": current_user.last_login
    }

# --- ROLE-SPECIFIC AI SCREENSHOT ANALYSIS ---
@app.post("/api/gameplay/analyze-screenshot")
async def analyze_screenshot(
    file: UploadFile = File(...),
    player_name: str = Form(...),
    role: str = Form(...),  # Tank, Mage, Assassin, Fighter, Core, Support, Marksman
    current_user: User = Depends(get_current_user)
):
    """Upload a post-game screenshot and get role-specific AI coaching report"""
    
    # 1. Validate File Type
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Only image files (JPG/PNG) are allowed.")
    
    # 2. Read and Check File Size (Increased to 10MB for mobile screenshots)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size must be less than 10MB. Please compress your screenshot.")
    
    # 3. Encode Image to Base64
    base64_image = base64.b64encode(contents).decode('utf-8')

    # 4. Role-specific coaching guidelines
    role_guidelines = {
        "Tank": "As a Tank, your job is to initiate fights, protect carries, and absorb damage. You should have high assists, low deaths, and defensive items. Map rotation is critical - you must be first to objectives.",
        "Mage": "As a Mage, your job is to deal burst magic damage, control lanes, and secure picks. You should have high kills, good gold income, and magic penetration items. Positioning is critical - stay behind tanks.",
        "Assassin": "As an Assassin, your job is to eliminate enemy carries and secure kills. You should have high kills, good gold income, and damage items. Map awareness is critical - look for isolated targets.",
        "Fighter": "As a Fighter, your job is to split push, duel enemies, and deal sustained damage. You should have balanced KDA, good gold, and hybrid items. Split pushing timing is critical.",
        "Core": "As a Core/Jungler, your job is to farm efficiently, secure objectives (Turtle/Lord), and gank lanes. You should have the highest gold on team, good KDA, and damage items. Farming efficiency is critical.",
        "Support": "As a Support, your job is to heal/shield carries, provide vision, and set up kills. You should have high assists, low deaths, and utility items. Vision control is critical.",
        "Marksman": "As a Marksman, your job is to farm safely early, deal sustained physical damage late game, and secure objectives. You should have high gold, good KDA, and attack items. Positioning is critical - never die."
    }

    role_meta_advice = {
        "Tank": "Current Meta Tanks: Tigreal, Atlas, Khufra. Meta builds focus on cooldown reduction and crowd control. Roam early, secure vision, and initiate team fights.",
        "Mage": "Current Meta Mages: Valentina, Yve, Pharsa. Meta builds focus on magic penetration and cooldown reduction. Control mid lane, rotate to side lanes, and burst enemies.",
        "Assassin": "Current Meta Assassins: Ling, Fanny, Hayabusa. Meta builds focus on critical chance and physical penetration. Farm jungle efficiently, look for picks, and eliminate carries.",
        "Fighter": "Current Meta Fighters: Yu Zhong, Paquito, Julian. Meta builds focus on hybrid damage and sustain. Split push when team is grouped, duel enemies, and pressure lanes.",
        "Core": "Current Meta Cores: Ling, Lancelot, Natalia. Meta builds focus on damage and mobility. Farm efficiently, secure Turtle/Lord, and gank overextended enemies.",
        "Support": "Current Meta Supports: Rafaela, Estes, Diggie. Meta builds focus on healing/shielding and cooldown reduction. Stay with carries, provide vision, and set up kills.",
        "Marksman": "Current Meta Marksmen: Brody, Wanwan, Beatrix. Meta builds focus on attack speed and critical damage. Farm safely early, position well in fights, and push towers."
    }

    guideline = role_guidelines.get(role, "Play your role effectively.")
    meta_advice = role_meta_advice.get(role, "Follow current meta builds.")

    # 5. AI System Prompt
    system_prompt = f"""You are the world's greatest Mobile Legends: Bang Bang Esports Coach specializing in {role} role analysis.
    
    CRITICAL RULES:
    1. NEVER use "N/A", "Data not available", or empty strings. Every field MUST have detailed content.
    2. DURATION: Look at the TOP RIGHT of the scoreboard. It says "Duration XX:XX". Extract this EXACTLY.
    3. RESULT: Look at the TOP CENTER. It says "VICTORY" or "DEFEAT" in large text. Extract this EXACTLY.
    4. HERO: If you can identify the hero portrait next to "{player_name}", name it. Otherwise use "Unknown {role} Hero".
    5. If you cannot find "{player_name}" on the scoreboard, return ONLY: {{"error": "Player not found", "message": "The name '{player_name}' does not appear on this scoreboard."}}
    6. Write EXTREMELY detailed analysis - minimum 400 words per section.
    7. Analyze based on {role} role: {guideline}
    8. Include current meta advice: {meta_advice}"""

    # 6. AI User Prompt
    user_prompt = f"""Analyze this Mobile Legends scoreboard for player: "{player_name}" who played {role} role.

CRITICAL EXTRACTION:
- Look at TOP RIGHT for "Duration XX:XX" - extract the exact time (e.g., "16:34")
- Look at TOP CENTER for "VICTORY" or "DEFEAT" - extract the result
- Find "{player_name}" by reading usernames next to hero portraits
- Extract their KDA (Kills/Deaths/Assists), Gold, all 6 Items, Score/Rating

Provide MASSIVE detailed analysis based on {role} role expectations.

Return ONLY valid JSON with this EXACT structure:
{{
    "match_summary": "Victory/Defeat | Duration: XX:XX | KDA: X/X/X | Gold: XXXX",
    "overall_rating": "X.X/10 - Detailed 1-sentence summary",
    "duration": "XX:XX",
    "result": "Victory or Defeat",
    "hero_played": "{role} (User selected {role} role)",
    "role_analysis": {{
        "title": "{role} Role Performance Analysis",
        "role_expectations": "{guideline}",
        "performance_score": "XX/100",
        "detailed_analysis": "Write 5-6 LONG paragraphs (600+ words) analyzing: 1) Did they fulfill their {role} role responsibilities? 2) KDA analysis - is it appropriate for a {role}? 3) Gold efficiency - did they farm enough for their role? 4) Score/rating compared to teammates. 5) Item build appropriateness for {role}. 6) Overall impact on the game. Cite specific numbers.",
        "strengths": [
            "Detailed strength #1 with evidence (2-3 sentences)",
            "Detailed strength #2 with evidence (2-3 sentences)",
            "Detailed strength #3 with evidence (2-3 sentences)",
            "Detailed strength #4 with evidence (2-3 sentences)",
            "Detailed strength #5 with evidence (2-3 sentences)"
        ],
        "weaknesses": [
            "Detailed weakness #1 with evidence (2-3 sentences)",
            "Detailed weakness #2 with evidence (2-3 sentences)",
            "Detailed weakness #3 with evidence (2-3 sentences)",
            "Detailed weakness #4 with evidence (2-3 sentences)",
            "Detailed weakness #5 with evidence (2-3 sentences)"
        ],
        "actionable_tips": [
            "Specific {role} tip #1 (2-3 sentences)",
            "Specific {role} tip #2 (2-3 sentences)",
            "Specific {role} tip #3 (2-3 sentences)",
            "Specific {role} tip #4 (2-3 sentences)",
            "Specific {role} tip #5 (2-3 sentences)"
        ]
    }},
    "mistakes_corrections": {{
        "title": "Critical Mistakes & Corrections",
        "critical_errors": [
            {{
                "mistake": "Specific mistake #1 for a {role} player (2-3 sentences)",
                "evidence": "Concrete evidence from screenshot with numbers (2-3 sentences)",
                "correction": "Detailed correction with specific in-game examples (4-5 sentences)",
                "analysis": "Why this mistake is especially bad for a {role} player (2-3 sentences)"
            }},
            {{
                "mistake": "Specific mistake #2 (2-3 sentences)",
                "evidence": "Evidence with numbers (2-3 sentences)",
                "correction": "Detailed correction (4-5 sentences)",
                "analysis": "Why this is bad for {role} (2-3 sentences)"
            }},
            {{
                "mistake": "Specific mistake #3 (2-3 sentences)",
                "evidence": "Evidence (2-3 sentences)",
                "correction": "Correction (4-5 sentences)",
                "analysis": "Analysis (2-3 sentences)"
            }},
            {{
                "mistake": "Specific mistake #4 (2-3 sentences)",
                "evidence": "Evidence (2-3 sentences)",
                "correction": "Correction (4-5 sentences)",
                "analysis": "Analysis (2-3 sentences)"
            }}
        ]
    }},
    "itemization_macro": {{
        "title": "Itemization & Macro Strategy",
        "overall_score": "XX/100",
        "items_built": ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6"],
        "detailed_analysis": "Write 5-6 LONG paragraphs (600+ words) analyzing: 1) Is this build optimal for {role}? 2) Do items counter the enemy team? 3) Item timing - were core items finished on time? 4) Missing key items for {role}. 5) Specific item swaps that would improve performance. 6) Comparison to pro {role} builds. Name exact items and explain why.",
        "macro_assessment": "Write 4-5 paragraphs (400+ words) analyzing: 1) Gold efficiency for {role}. 2) Did they farm enough? 3) Objective control based on duration/result. 4) Map awareness and rotation patterns. 5) What to prioritize in future games.",
        "actionable_tips": [
            "Specific item tip #1 for {role} (2-3 sentences)",
            "Specific item tip #2 for {role} (2-3 sentences)",
            "Specific macro tip #3 (2-3 sentences)",
            "Specific macro tip #4 (2-3 sentences)",
            "Specific macro tip #5 (2-3 sentences)"
        ]
    }},
    "meta_recommendations": {{
        "title": "Current Meta Advice for {role}",
        "meta_heroes": "List 3-5 current meta {role} heroes and why they're strong (4-5 sentences)",
        "meta_builds": "Describe the current meta build path for {role} heroes (4-5 sentences)",
        "meta_strategy": "Explain current meta strategy for {role} players - when to rotate, when to fight, when to farm (5-6 sentences)",
        "meta_items": "List 3-4 must-have meta items for {role} and when to build them (4-5 sentences)",
        "rank_up_tips": "Specific tips to climb ranks playing {role} in current meta (5-6 sentences)"
    }},
    "overall_recommendations": {{
        "priority_1": "The #1 thing to fix (3-4 detailed sentences)",
        "priority_2": "The #2 thing (3-4 sentences)",
        "priority_3": "The #3 thing (3-4 sentences)",
        "long_term_goal": "What rank they can reach with timeline and milestones (4-5 sentences)"
    }}
}}

REMEMBER: Extract Duration from top right (e.g., "16:34"). Extract Result from top center (VICTORY/DEFEAT). Every field must be filled. NO EMPTY STRINGS. NO N/A. Write extensive paragraphs."""

    # 7. Call OpenAI API
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
            max_tokens=4500
        )
        
        content = response.choices[0].message.content
        print("RAW AI RESPONSE LENGTH:", len(content))
        
        # Clean markdown code blocks if present
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        
        analysis_data = json.loads(content)
        
        # Check for error response (Player not found)
        if "error" in analysis_data and analysis_data["error"] == "Player not found":
            raise HTTPException(status_code=404, detail=analysis_data["message"])

        # Add metadata
        analysis_data["file_id"] = f"screenshot_{current_user.id}_{datetime.utcnow().timestamp()}"
        analysis_data["player_focus"] = player_name
        analysis_data["role_focus"] = role
        
        return analysis_data

    except json.JSONDecodeError as e:
        print("JSON DECODE ERROR")
        print("Raw content:", content[:500])
        raise HTTPException(status_code=500, detail="AI returned invalid JSON. Please try a clearer screenshot.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ANALYSIS ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# --- SUBSCRIPTION UPGRADE ---
@app.post("/api/subscription/upgrade")
async def upgrade_subscription(
    request: UpgradeRequest, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Called by frontend after successful PayPal payment"""
    plan = request.plan_name.lower()
    
    if plan not in ["pro", "elite"]:
        raise HTTPException(status_code=400, detail="Invalid plan name. Must be 'pro' or 'elite'")
    
    current_user.is_premium = True
    current_user.plan_type = plan
    db.commit()
    
    return {
        "message": f"Successfully upgraded to {plan}!",
        "is_premium": True,
        "plan_type": plan
    }

# --- AVATAR UPLOAD ---
@app.post("/api/user/avatar")
async def upload_avatar(
    file: UploadFile = File(...), 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload custom avatar image"""
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images allowed")
    
    contents = await file.read()
    if len(contents) > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be less than 2MB")
    
    upload_dir = Path("uploads/avatars")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / f"{current_user.id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(contents)
    
    avatar_url = f"/uploads/avatars/{current_user.id}_{file.filename}"
    current_user.avatar_url = avatar_url
    db.commit()
    
    return {"avatar_url": avatar_url, "message": "Avatar updated successfully"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "mythicvision-backend"}

# --- SERVER STARTUP (Required for Render) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
