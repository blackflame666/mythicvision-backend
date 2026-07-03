import os
import shutil
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
import uvicorn

# Load environment variables
load_dotenv()

# --- APP INITIALIZATION ---
app = FastAPI(title="MythicVision Backend - ML Coach")

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
class AnalyzeRequest(BaseModel):
    hero_name: str = None

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

# --- GAMEPLAY ANALYSIS (Premium Feature) ---
@app.post("/api/gameplay/analyze")
async def analyze_gameplay(request: AnalyzeRequest, current_user: User = Depends(get_current_user)):
    """Analyze gameplay. Hero selection is Premium only."""
    if request.hero_name and not current_user.is_premium:
        raise HTTPException(
            status_code=403, 
            detail="Selecting a specific hero is a Premium feature. Please upgrade your subscription."
        )
    
    focus_text = f"focusing specifically on {request.hero_name}" if request.hero_name else "analyzing overall gameplay"
    return {
        "message": "Analysis started",
        "user": current_user.name,
        "is_premium": current_user.is_premium or False,
        "plan_type": current_user.plan_type or "free",
        "analysis_focus": focus_text,
        "status": "processing video..."
    }

# --- VIDEO UPLOAD ENDPOINT ---
@app.post("/api/gameplay/upload")
async def upload_gameplay(
    file: UploadFile = File(...),
    hero_name: str = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload MLBB match replay for AI analysis"""
    
    # Validate file type
    allowed_types = ["video/mp4", "video/quicktime", "video/x-msvideo"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only MP4, MOV, and AVI video files allowed")
    
    # Check file size (500MB max)
    contents = await file.read()
    if len(contents) > 500 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be less than 500MB")
    
    # Check if user is premium for hero selection
    if hero_name and hero_name.lower() != "null" and not current_user.is_premium:
        raise HTTPException(
            status_code=403,
            detail="Hero-specific analysis is an Elite feature. Please upgrade your subscription."
        )
    
    # Create uploads directory
    upload_dir = Path("uploads/gameplay")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file with unique name
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'mp4'
    unique_filename = f"{current_user.id}_{datetime.utcnow().timestamp()}.{file_extension}"
    file_path = upload_dir / unique_filename
    
    with open(file_path, "wb") as buffer:
        buffer.write(contents)
    
    # Return success response
    return {
        "message": "Video uploaded successfully",
        "file_id": unique_filename,
        "hero_focus": hero_name if hero_name and hero_name.lower() != "null" else "overall gameplay",
        "status": "queued for analysis",
        "user": current_user.name,
        "plan_type": current_user.plan_type
    }

# --- SUBSCRIPTION UPGRADE (PayPal Integration) ---
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
    
    # Check file size (2MB max)
    contents = await file.read()
    if len(contents) > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be less than 2MB")
    
    # Create uploads directory
    upload_dir = Path("uploads/avatars")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = upload_dir / f"{current_user.id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(contents)
    
    # Update user avatar URL
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
