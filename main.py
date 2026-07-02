import os
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from authlib.integrations.starlette_client import OAuth
from jose import jwt, JWTError
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Load environment variables
load_dotenv()

app = FastAPI(title="MythicVision Backend - ML Coach")

# --- DATABASE SETUP (Simple SQLAlchemy) ---
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mlcoach.db")
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    avatar_url = Column(String)
    google_id = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- CORS & SECURITY ---
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://mlcoach.online")
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-change-this-in-production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000"], # Allow local dev too
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        'redirect_uri': f"{os.getenv('API_URL', 'https://api.mlcoach.online')}/auth/google/callback"
    },
)

# --- AUTH ROUTES ---

@app.get("/auth/google/login")
async def google_login(request: Request):
    """Step 1: Redirect user to Google"""
    redirect_uri = request.url_for('google_callback')
    # If running locally, you might need to hardcode this to your production callback URL
    # redirect_uri = "https://api.mlcoach.online/auth/google/callback" 
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    """Step 2: Google redirects back here with user info"""
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        
        if not user_info:
            raise HTTPException(status_code=400, detail="Failed to get user info from Google")

        email = user_info.get('email')
        name = user_info.get('name')
        avatar = user_info.get('picture')
        google_id = user_info.get('sub')

        # Check if user exists in DB, if not, create them
        db_user = db.query(User).filter(User.email == email).first()
        if not db_user:
            db_user = User(email=email, name=name, avatar_url=avatar, google_id=google_id)
            db.add(db_user)
            db.commit()
            db.refresh(db_user)

        # Generate JWT Token
        token_data = {"sub": email, "user_id": db_user.id, "name": name}
        access_token = jwt.encode(token_data, SECRET_KEY, algorithm="HS256")

        # Redirect back to Frontend Dashboard with the token in the URL 
        # (Frontend will grab this and store it)
        return RedirectResponse(url=f"{FRONTEND_URL}/dashboard?token={access_token}")

    except Exception as e:
        print(f"Auth Error: {e}")
        raise HTTPException(status_code=400, detail="Authentication failed")

# --- PROTECTED API ROUTES ---

@app.get("/api/me")
async def get_current_user(request: Request, db: Session = Depends(get_db)):
    """Get the currently logged-in user"""
    token = request.cookies.get("access_token") or request.headers.get("Authorization")
    
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if token.startswith("Bearer "):
        token = token.split(" ")[1]

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload.get("sub")
        user = db.query(User).filter(User.email == email).first()
        return {"id": user.id, "email": user.email, "name": user.name, "avatar": user.avatar_url}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/")
def root():
    return {"message": "MythicVision API is running! Go to /auth/google/login to start."}
