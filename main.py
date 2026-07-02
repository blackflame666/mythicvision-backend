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
    avatar_url = Column(String)
    google_id = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- CORS & SECURITY CONFIG ---
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://mlcoach.online")
API_URL = os.getenv("API_URL", "https://mythicvision-backend.onrender.com")
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-change-this-in-production")
ALGORITHM = "HS256"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://localhost:8000"],
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
        'redirect_uri': f"{API_URL}/auth/google/callback"
    },
)

# --- HELPER FUNCTIONS ---
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=7))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    # Check for token in Authorization header or cookies
    token = request.headers.get("Authorization") or request.cookies.get("access_token")
    
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
    return {
        "message": "MythicVision API is running!",
        "docs": "/docs",
        "login": "/auth/google/login"
    }

@app.get("/auth/google/login")
async def google_login(request: Request):
    """Step 1: Redirect user to Google for authentication"""
    redirect_uri = request.url_for('google_callback')
    # If running locally, you might need to override this to your production callback URL
    # redirect_uri = f"{API_URL}/auth/google/callback"
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
        else:
            # Update last login time
            db_user.last_login = datetime.utcnow()
            db.commit()

        # Generate JWT Token
        token_data = {"sub": email, "user_id": db_user.id, "name": db_user.name}
        access_token = create_access_token(data=token_data)

        # Redirect back to Frontend Dashboard with the token in the URL 
        # The frontend should grab this token and store it in localStorage
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
        "created_at": current_user.created_at,
        "last_login": current_user.last_login
    }

@app.post("/api/gameplay/analyze")
async def analyze_gameplay(current_user: User = Depends(get_current_user)):
    """Protected route: Analyze gameplay (Placeholder for future AI logic)"""
    return {
        "message": "Gameplay analysis endpoint ready",
        "user": current_user.name,
        "status": "awaiting video upload"
    }

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

# --- SERVER STARTUP (Required for Render) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
