import os
import json
import re
import base64
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from jose import jwt, JWTError
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional, List
import uvicorn

# Load environment variables
load_dotenv()

# --- APP INITIALIZATION ---
app = FastAPI(title="MythicVision Backend - ML Coach & Tournament")

# --- OPENAI SETUP ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set. AI analysis will fail.")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- DATABASE SETUP ---
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
   # Create tables if they don't exist
Base.metadata.create_all(bind=engine)
   

# --- DATABASE MODELS ---

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String)
    avatar_url = Column(String, nullable=True)
    google_id = Column(String, unique=True, nullable=True)
    is_premium = Column(Boolean, default=False)
    plan_type = Column(String, default="free")  # "free", "pro", or "elite"
    is_admin = Column(Boolean, default=False)  # Admin flag
    subscription_end_date = Column(DateTime, nullable=True)  # 30-day timer
    hashed_password = Column(String, nullable=True)  # For email/password auth
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tournaments = relationship("Tournament", back_populates="creator")

class Tournament(Base):
    __tablename__ = "tournaments"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    max_teams = Column(Integer, default=8)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    start_date = Column(DateTime)
    game_mode = Column(String)
    description = Column(String, nullable=True)
    prize_pool = Column(String, nullable=True)  # <--- Make sure this exists!
    
    # Relationships
    creator = relationship("User", back_populates="tournaments")
    teams = relationship("TournamentTeam", back_populates="tournament", cascade="all, delete-orphan")
    matches = relationship("TournamentMatch", back_populates="tournament", cascade="all, delete-orphan")

class TournamentTeam(Base):
    __tablename__ = "tournament_teams"
    id = Column(Integer, primary_key=True, index=True)
    tournament_id = Column(Integer, ForeignKey("tournaments.id"), nullable=False)
    team_name = Column(String, nullable=False)
    captain_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    player_ids = Column(String)  # JSON string of user IDs
    seed = Column(Integer)  # Tournament seeding
    logo_url = Column(String, nullable=True)
    
    # Relationships
    tournament = relationship("Tournament", back_populates="teams")

class TournamentMatch(Base):
    __tablename__ = "tournament_matches"
    id = Column(Integer, primary_key=True, index=True)
    tournament_id = Column(Integer, ForeignKey("tournaments.id"), nullable=False)
    round_number = Column(Integer)  # 1 = quarterfinal, 2 = semifinal, etc.
    team1_id = Column(Integer, ForeignKey("tournament_teams.id"))
    team2_id = Column(Integer, ForeignKey("tournament_teams.id"))
    winner_id = Column(Integer, ForeignKey("tournament_teams.id"))
    match_order = Column(Integer)  # Position in bracket
    status = Column(String, default="pending")  # pending, completed, scheduled
    scheduled_time = Column(DateTime)
    team1_score = Column(Integer, default=0)
    team2_score = Column(Integer, default=0)
    
    # Relationships
    tournament = relationship("Tournament", back_populates="matches")

# Create tables
Base.metadata.create_all(bind=engine)

# Database migration - add new columns to existing users table
# Database migration - add new columns
with engine.connect() as conn:
    try:
        # Add columns to users table
        conn.execute(text("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0"))
        conn.execute(text("ALTER TABLE users ADD COLUMN subscription_end_date DATETIME"))
        conn.execute(text("ALTER TABLE users ADD COLUMN hashed_password VARCHAR"))
        
        # Add column to tournaments table
        conn.execute(text("ALTER TABLE tournaments ADD COLUMN prize_pool VARCHAR"))
        
        conn.commit()
        print("Added migration columns successfully")
    except Exception as e:
        print(f"Migration info (columns may already exist): {e}")
        conn.rollback()

# --- AUTO-PROMOTE ADMIN (Run once on deploy) ---
with Session(engine) as db:
    try:
        admin_email = "delram540@gmail.com"
        user = db.query(User).filter(User.email == admin_email).first()
        if user and not user.is_admin:
            user.is_admin = True
            user.plan_type = "elite"
            user.is_premium = True
            user.subscription_end_date = datetime.utcnow() + timedelta(days=365) # 1 year free for you!
            db.commit()
            print(f"✅ AUTO-PROMOTED {admin_email} to Admin & Elite!")
    except Exception as e:
        print(f"Auto-promote skipped (user might not exist yet): {e}")
# -------------------------------------------------

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

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class TournamentCreateRequest(BaseModel):
    name: str
    max_teams: int
    game_mode: str
    start_date: datetime
    description: Optional[str] = None
    prize_pool: Optional[str] = None  # <--- Make sure this line exists

class TeamRegisterRequest(BaseModel):
    team_name: str
    captain_id: int
    player_ids: List[int]

class MatchResultRequest(BaseModel):
    winner_id: int
    team1_score: int
    team2_score: int

class AdminUpdateRequest(BaseModel):
    user_email: str
    plan_type: str  # "free", "pro", "elite"
    days: int = 30  # Default 30 days

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
        
        # Check subscription expiry
        if user.subscription_end_date and user.subscription_end_date < datetime.utcnow():
            user.plan_type = "free"
            user.is_premium = False
            user.subscription_end_date = None
            db.commit()
        
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def generate_bracket_matches(tournament_id: int, num_teams: int, db: Session):
    """Generate tournament bracket matches based on number of teams"""
    teams = db.query(TournamentTeam).filter(TournamentTeam.tournament_id == tournament_id).all()
    
    # Calculate number of rounds
    import math
    num_rounds = int(math.log2(num_teams))
    
    # Create matches for each round
    matches = []
    
    # Round 1 (Quarterfinals/Semifinals depending on size)
    for i in range(0, len(teams), 2):
        if i + 1 < len(teams):
            match = TournamentMatch(
                tournament_id=tournament_id,
                round_number=1,
                team1_id=teams[i].id,
                team2_id=teams[i + 1].id,
                match_order=i // 2,
                status="pending"
            )
            matches.append(match)
    
    # Create empty slots for subsequent rounds
    for round_num in range(2, num_rounds + 1):
        matches_in_round = num_teams // (2 ** round_num)
        for i in range(matches_in_round):
            match = TournamentMatch(
                tournament_id=tournament_id,
                round_number=round_num,
                team1_id=None,
                team2_id=None,
                match_order=i,
                status="pending"
            )
            matches.append(match)
    
    db.add_all(matches)
    db.commit()
    return matches

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
            # AUTO-PROMOTE ADMIN FIX
            is_admin = (email == "delram540@gmail.com")
            db_user = User(
                email=email, 
                name=name, 
                avatar_url=avatar, 
                google_id=google_id, 
                is_premium=is_admin,
                plan_type="elite" if is_admin else "free",
                is_admin=is_admin,
                subscription_end_date=datetime.utcnow() + timedelta(days=365) if is_admin else None
            )
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
        else:
            # Ensure admin stays admin
            if email == "delram540@gmail.com":
                db_user.is_admin = True
                db_user.plan_type = "elite"
                db_user.is_premium = True
                db.commit()
            
            db_user.last_login = datetime.utcnow()
            db.commit()

        # Generate JWT Token
        access_token = create_access_token(data={"sub": email, "user_id": db_user.id, "name": db_user.name})
        
        # CRITICAL FIX: Redirect to /login WITH token parameter
        redirect_url = f"{FRONTEND_URL}/login?token={access_token}"
        print(f"✅ Redirecting to: {redirect_url}")
        return RedirectResponse(url=redirect_url)
    except Exception as e:
        print(f"Auth Error: {e}")
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

# --- EMAIL/PASSWORD AUTH ENDPOINTS ---
@app.post("/api/auth/register")
async def register_user(
    request: RegisterRequest,
    db: Session = Depends(get_db)
):
    """Register a new user with email and password"""
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    hashed_password = pwd_context.hash(request.password)
    
    # Create new user with AUTO-PROMOTE ADMIN FIX
    is_admin = (request.email == "delram540@gmail.com")
    new_user = User(
        email=request.email,
        name=request.name,
        google_id=None,
        avatar_url=None,
        is_premium=is_admin,
        plan_type="elite" if is_admin else "free",
        is_admin=is_admin,
        hashed_password=hashed_password,
        subscription_end_date=datetime.utcnow() + timedelta(days=365) if is_admin else None
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Generate JWT token
    access_token = create_access_token(
        data={"sub": new_user.email, "user_id": new_user.id, "name": new_user.name}
    )
    
    return {
        "message": "User created successfully",
        "access_token": access_token,
        "user": {
            "id": new_user.id,
            "email": new_user.email,
            "name": new_user.name,
            "plan_type": new_user.plan_type
        }
    }

@app.post("/api/auth/login")
async def login_user(
    request: LoginRequest,
    db: Session = Depends(get_db)
):
    """Login with email and password"""
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Check if user has a password (Google users won't)
    if not user.hashed_password:
        raise HTTPException(status_code=401, detail="Please login with Google for this account")
    
       # Verify password
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    if not pwd_context.verify(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
     # AUTO-PROMOTE ADMIN FIX ON LOGIN
    if user.email == "delram540@gmail.com":
        user.is_admin = True
        user.plan_type = "elite"
        user.is_premium = True
        user.subscription_end_date = datetime.utcnow() + timedelta(days=365)
        db.commit()
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Generate JWT token
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id, "name": user.name}
    )
    
    return {
        "message": "Login successful",
        "access_token": access_token,
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "plan_type": user.plan_type
        }
    }

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
        "is_admin": current_user.is_admin or False,
        "subscription_end_date": current_user.subscription_end_date,
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

    # 7. Call OpenAI API with Retry Logic
    max_retries = 3
    for attempt in range(max_retries):
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
                max_tokens=3000
            )
            break  # If successful, break out of the retry loop
            
        except Exception as e:
            print(f"OpenAI Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(3)  # Wait 3 seconds before retrying
            else:
                print(f"CRITICAL ANALYSIS ERROR after {max_retries} attempts: {e}")
                raise HTTPException(status_code=500, detail="AI is temporarily busy. Please try again in a minute.")
    
    try:
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

# --- TOURNAMENT ENDPOINTS (ELITE EXCLUSIVE) ---

@app.post("/api/tournaments/create")
async def create_tournament(
    tournament_data: TournamentCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new tournament (ELITE EXCLUSIVE)"""
    # STRICT CHECK: Only Elite users can create tournaments
    if current_user.plan_type != "elite":
        raise HTTPException(status_code=403, detail="Tournament creation is an Elite-exclusive feature. Please upgrade to Elite.")
    
    tournament = Tournament(
        name=tournament_data.name,
        creator_id=current_user.id,
        max_teams=tournament_data.max_teams,
        game_mode=tournament_data.game_mode,
        start_date=tournament_data.start_date,
        description=tournament_data.description,
        prize_pool=tournament_data.prize_pool,  # <--- THIS LINE MUST BE HERE
        status="pending"
    )
    
    db.add(tournament)
    db.commit()
    db.refresh(tournament)
    
    return {
        "message": "Tournament created successfully",
        "tournament_id": tournament.id,
        "tournament": tournament
    }

@app.get("/api/tournaments")
async def get_user_tournaments(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all tournaments created by user"""
    from sqlalchemy.orm import joinedload
    tournaments = db.query(Tournament).options(joinedload(Tournament.teams)).filter(Tournament.creator_id == current_user.id).all()
    return {"tournaments": tournaments}

@app.get("/api/tournaments/{tournament_id}")
async def get_tournament_bracket(
    tournament_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get tournament details with bracket and teams (All authenticated users can view)"""
    tournament = db.query(Tournament).filter(Tournament.id == tournament_id).first()
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")
    
    teams = db.query(TournamentTeam).filter(TournamentTeam.tournament_id == tournament_id).all()
    matches = db.query(TournamentMatch).filter(TournamentMatch.tournament_id == tournament_id).order_by(
        TournamentMatch.round_number, TournamentMatch.match_order
    ).all()
    
    return {
        "tournament": tournament,
        "teams": teams,
        "matches": matches
    }

@app.post("/api/tournaments/{tournament_id}/register")
async def register_team(
    tournament_id: int,
    team_data: TeamRegisterRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Register a team for a tournament (CREATOR ONLY)"""
    tournament = db.query(Tournament).filter(Tournament.id == tournament_id).first()
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")
    
    # CRITICAL CHECK: Only the tournament creator can register teams
    if tournament.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the tournament creator can register teams.")
    
    if tournament.status != "pending":
        raise HTTPException(status_code=400, detail="Tournament is no longer accepting registrations")
    
    # Check if tournament is full
    current_teams = db.query(TournamentTeam).filter(TournamentTeam.tournament_id == tournament_id).count()
    if current_teams >= tournament.max_teams:
        raise HTTPException(status_code=400, detail="Tournament is full")
    
    team = TournamentTeam(
        tournament_id=tournament_id,
        team_name=team_data.team_name,
        captain_id=current_user.id,
        player_ids=json.dumps([current_user.id]),
        seed=current_teams + 1
    )
    
    db.add(team)
    db.commit()
    db.refresh(team)
    
    return {"message": "Team registered successfully", "team": team}

@app.post("/api/tournaments/{tournament_id}/start")
async def start_tournament(
    tournament_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start tournament and generate bracket (ELITE EXCLUSIVE)"""
    tournament = db.query(Tournament).filter(Tournament.id == tournament_id).first()
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")
    
    if tournament.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only tournament creator can start the tournament")
    
    # STRICT CHECK: Only Elite users can start/manage tournaments
    if current_user.plan_type != "elite":
        raise HTTPException(status_code=403, detail="Tournament management is an Elite-exclusive feature. Please upgrade to Elite.")
    
    if tournament.status != "pending":
        raise HTTPException(status_code=400, detail="Tournament has already started")
    
    # Check if we have enough teams
    num_teams = db.query(TournamentTeam).filter(TournamentTeam.tournament_id == tournament_id).count()
    if num_teams < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 teams to start tournament")
    
    # Generate bracket matches
    generate_bracket_matches(tournament_id, tournament.max_teams, db)
    
    tournament.status = "active"
    db.commit()
    
    return {"message": "Tournament started successfully", "bracket_generated": True}

@app.delete("/api/tournaments/{tournament_id}")
async def delete_tournament(
    tournament_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a tournament (CREATOR ONLY)"""
    tournament = db.query(Tournament).filter(Tournament.id == tournament_id).first()
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")
    
    if tournament.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the creator can delete this tournament")
    
    db.delete(tournament)
    db.commit()
    
    return {"message": "Tournament deleted successfully"}
 
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
    current_user.subscription_end_date = datetime.utcnow() + timedelta(days=30)
    db.commit()
    
    return {
        "message": f"Successfully upgraded to {plan} for 30 days!",
        "is_premium": True,
        "plan_type": plan,
        "subscription_end_date": current_user.subscription_end_date
    }

# --- ADMIN PANEL ---
@app.post("/api/admin/update-user")
async def admin_update_user(
    request: AdminUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Grant or revoke Pro/Elite status (ADMIN ONLY)"""
    # SECURITY CHECK: Only allow specific admin email OR users with is_admin flag
    if current_user.email != "delram540@gmail.com" and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied. Admins only.")
    
    target_user = db.query(User).filter(User.email == request.user_email).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found. They must log in first.")
    
    # Update Plan
    target_user.plan_type = request.plan_type
    target_user.is_premium = request.plan_type in ["pro", "elite"]
    
    # Update Expiry Date
    if request.plan_type == "free":
        target_user.subscription_end_date = None
    else:
        target_user.subscription_end_date = datetime.utcnow() + timedelta(days=request.days)
    
    db.commit()
    
    return {
        "message": f"Successfully updated {target_user.email} to {request.plan_type} for {request.days} days.",
        "new_expiry": target_user.subscription_end_date
    }

@app.get("/api/admin/users")
async def admin_list_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all users in the database (ADMIN ONLY)"""
    # SECURITY CHECK
    if current_user.email != "delram540@gmail.com" and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied. Admins only.")
    
    users = db.query(User).all()
    
    return {
        "total_users": len(users),
        "users": [
            {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "plan_type": user.plan_type,
                "is_admin": user.is_admin,
                "subscription_end_date": str(user.subscription_end_date) if user.subscription_end_date else None,
                "created_at": str(user.created_at)
            }
            for user in users
        ]
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

@app.post("/api/tournaments/matches/{match_id}/result")
async def submit_match_result(
    match_id: int,
    result_data: MatchResultRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit match result and advance winner"""
    match = db.query(TournamentMatch).filter(TournamentMatch.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    match.winner_id = result_data.winner_id
    match.team1_score = result_data.team1_score
    match.team2_score = result_data.team2_score
    match.status = "completed"
    
    if match.round_number < 3:
        next_round_matches = db.query(TournamentMatch).filter(
            TournamentMatch.tournament_id == match.tournament_id,
            TournamentMatch.round_number == match.round_number + 1
        ).all()
        
        for next_match in next_round_matches:
            if not next_match.team1_id:
                next_match.team1_id = result_data.winner_id
                break
            elif not next_match.team2_id:
                next_match.team2_id = result_data.winner_id
                break
    
    db.commit()
    return {"message": "Match result submitted successfully"}
