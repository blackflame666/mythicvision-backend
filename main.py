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
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

class UpgradeRequest(BaseModel):
    plan_name: str

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
    return await oauth.google.authorize_redirect(request, f"{API_URL}/auth/google/callback")

@app.get("/auth/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
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
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "avatar_url": current_user.avatar_url,
        "is_premium": current_user.is_premium or False,
        "plan_type": current_user.plan_type or "free",
        "created_at": current_user.created_at,
        "last_login": current_user.last_login
    }

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

    # THE ULTIMATE COACH PROMPT - FORCES LONG, DETAILED OUTPUT
    system_prompt = """You are the world's greatest Mobile Legends: Bang Bang Esports Coach. You provide EXTREMELY DETAILED, lengthy analysis. Each section must be 300-500 words minimum. You never return N/A, empty strings, or 'No details provided'. You are brutally honest, specific, and actionable."""

    user_prompt = f"""Analyze this Mobile Legends post-game scoreboard for the player: "{player_name}"

CRITICAL INSTRUCTIONS:
1. Find the player "{player_name}" on the scoreboard by reading the text usernames next to hero portraits.
2. Extract: Hero (identify from portrait), KDA (Kills/Deaths/Assists), Gold, all 6 Items, Score/Rating, Team side (left or right).
3. Determine: Win/Loss (from the big VICTORY/DEFEAT text at top), Game Duration (from top right).
4. Analyze EVERY aspect in EXTREME DETAIL - write 3-4 paragraphs per section minimum.

YOU MUST FILL IN EVERY SINGLE FIELD BELOW. NO "N/A", NO EMPTY STRINGS, NO "No details provided", NO "No additional details provided".
If you cannot determine something from the image, make an educated inference based on the visible stats and explain your reasoning.

Return ONLY valid JSON in this EXACT structure (use double quotes for all strings):

{{
    "match_summary": "Victory/Defeat | Duration: XX:XX | KDA: X/X/X | Gold: XXXX",
    "overall_rating": "X.X/10 - Detailed 1-sentence summary of performance",
    "duration": "XX:XX",
    "result": "Victory or Defeat",
    "hero_played": "Hero Name identified from portrait",
    "analysis": {{
        "gameplay_mechanics": {{
            "title": "Gameplay & Mechanics Deep Dive",
            "overall_score": "XX/100",
            "detailed_analysis": "Write 4-5 LONG paragraphs (500+ words total) analyzing: 1) Their KDA performance and what it reveals about their mechanics - were they getting picks, surviving fights, or feeding? 2) Their farming efficiency based on gold earned compared to teammates - did they maximize their economy? 3) Their score/rating and what it means in context of the match. 4) Comparison to teammates - were they carrying or being carried? 5) Role-specific expectations - as a [hero role], did they fulfill their job? Be brutally honest and cite specific numbers from the screenshot.",
            "strengths": [
                "Detailed strength #1 with specific evidence from screenshot (2-3 sentences)",
                "Detailed strength #2 with specific evidence (2-3 sentences)",
                "Detailed strength #3 with specific evidence (2-3 sentences)",
                "Detailed strength #4 with specific evidence (2-3 sentences)"
            ],
            "weaknesses": [
                "Detailed weakness #1 with specific evidence (2-3 sentences)",
                "Detailed weakness #2 with specific evidence (2-3 sentences)",
                "Detailed weakness #3 with specific evidence (2-3 sentences)",
                "Detailed weakness #4 with specific evidence (2-3 sentences)"
            ],
            "actionable_tips": [
                "Specific tip #1: Exactly what to practice and how, with a concrete example (2-3 sentences)",
                "Specific tip #2: Exactly what to practice and how, with a concrete example (2-3 sentences)",
                "Specific tip #3: Exactly what to practice and how, with a concrete example (2-3 sentences)",
                "Specific tip #4: Exactly what to practice and how, with a concrete example (2-3 sentences)",
                "Specific tip #5: Exactly what to practice and how, with a concrete example (2-3 sentences)"
            ]
        }},
        "mistakes_corrections": {{
            "title": "Critical Mistakes & The Why",
            "critical_errors": [
                {{
                    "mistake": "Specific mistake #1 - describe exactly what went wrong based on the stats (2-3 sentences)",
                    "evidence": "Concrete evidence from the screenshot - cite specific numbers like deaths, gold, items, or score that prove this mistake happened (2-3 sentences)",
                    "correction": "Detailed correction - explain EXACTLY what they should do differently next time, with specific in-game examples and scenarios (4-5 sentences)"
                }},
                {{
                    "mistake": "Specific mistake #2 (2-3 sentences)",
                    "evidence": "Concrete evidence from screenshot (2-3 sentences)",
                    "correction": "Detailed correction (4-5 sentences)"
                }},
                {{
                    "mistake": "Specific mistake #3 (2-3 sentences)",
                    "evidence": "Concrete evidence from screenshot (2-3 sentences)",
                    "correction": "Detailed correction (4-5 sentences)"
                }},
                {{
                    "mistake": "Specific mistake #4 (2-3 sentences)",
                    "evidence": "Concrete evidence from screenshot (2-3 sentences)",
                    "correction": "Detailed correction (4-5 sentences)"
                }}
            ]
        }},
        "itemization_macro": {{
            "title": "Itemization & Macro Strategy Analysis",
            "overall_score": "XX/100",
            "items_built": ["Item 1 name", "Item 2 name", "Item 3 name", "Item 4 name", "Item 5 name", "Item 6 name"],
            "detailed_analysis": "Write 4-5 LONG paragraphs (500+ words) analyzing: 1) Whether their item build is optimal for their specific hero - compare to standard pro builds. 2) Whether items counter the enemy team composition visible on the scoreboard - did they build anti-heal against healers, magic defense against mages, etc.? 3) Item timing and progression - did they finish core items at the right time? 4) Missing key items they should have built instead. 5) Specific item swaps that would have improved their performance. Be extremely specific - name exact items and explain why each is good or bad.",
            "macro_assessment": "Write 3-4 paragraphs (300+ words) analyzing: 1) Their gold efficiency - did they convert farm into impact? 2) Whether they farmed enough for their role - compare gold to teammates in similar roles. 3) Objective control inferred from game duration and result - did they play for early game or late game? 4) Map awareness and rotation patterns inferred from gold/assists ratio. 5) What they should prioritize in future games to improve macro play.",
            "actionable_tips": [
                "Specific item tip #1 - name exact items and when to build them (2-3 sentences)",
                "Specific item tip #2 - name exact items and when to build them (2-3 sentences)",
                "Specific macro tip #3 - specific in-game behavior to change (2-3 sentences)",
                "Specific macro tip #4 - specific in-game behavior to change (2-3 sentences)",
                "Specific macro tip #5 - specific in-game behavior to change (2-3 sentences)"
            ]
        }}
    }},
    "overall_recommendations": {{
        "priority_1": "The #1 most critical thing to fix - explain WHY it matters and HOW to fix it with specific steps (3-4 detailed sentences)",
        "priority_2": "The #2 most critical thing - explain WHY and HOW (3-4 detailed sentences)",
        "priority_3": "The #3 most critical thing - explain WHY and HOW (3-4 detailed sentences)",
        "long_term_goal": "Detailed 4-5 sentence analysis of what rank they can reach if they fix these issues, with a realistic timeline, specific milestones to hit, and encouragement"
    }}
}}

REMEMBER: Every field must be filled with substantial content. Every paragraph must be lengthy and detailed. NO SHORT ANSWERS. NO N/A. NO EMPTY STRINGS. This is a professional coaching report that a paying user deserves."""

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
        print("RAW AI RESPONSE START:", content[:500])
        print("RAW AI RESPONSE LENGTH:", len(content))
        
        # Clean markdown code blocks
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        
        analysis_data = json.loads(content)
        
        # Validate critical fields are not empty
        required_fields = [
            'match_summary',
            'overall_rating',
            'analysis.gameplay_mechanics.detailed_analysis',
            'analysis.mistakes_corrections.critical_errors',
            'analysis.itemization_macro.detailed_analysis'
        ]
        
        for field in required_fields:
            keys = field.split('.')
            obj = analysis_data
            for key in keys:
                if isinstance(obj, dict):
                    obj = obj.get(key)
                else:
                    obj = None
                    break
            if not obj or (isinstance(obj, str) and obj.strip() == ''):
                print(f"WARNING: Empty field detected: {field}")
        
        analysis_data["file_id"] = f"screenshot_{current_user.id}_{datetime.utcnow().timestamp()}"
        analysis_data["player_focus"] = player_name
        
        return analysis_data

    except json.JSONDecodeError as e:
        print("JSON DECODE ERROR")
        print("Raw content:", content)
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {str(e)}")
    except Exception as e:
        print(f"ANALYSIS ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"AI Analysis failed: {str(e)}")

# --- SUBSCRIPTION UPGRADE ---
@app.post("/api/subscription/upgrade")
async def upgrade_subscription(
    request: UpgradeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    plan = request.plan_name.lower()
    if plan not in ["pro", "elite"]:
        raise HTTPException(status_code=400, detail="Invalid plan name.")
    current_user.is_premium = True
    current_user.plan_type = plan
    db.commit()
    return {"message": f"Successfully upgraded to {plan}!", "is_premium": True, "plan_type": plan}

# --- AVATAR UPLOAD ---
@app.post("/api/user/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
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
    return {"status": "healthy", "service": "mythicvision-backend"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
