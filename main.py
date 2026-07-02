from fastapi import APIRouter, Request, HTTPException
from authlib.integrations.starlette_client import OAuth
from jose import jwt
import os

router = APIRouter()
oauth = OAuth()

# Configure Google OAuth
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

@router.get("/auth/google/login")
async def google_login(request: Request):
    # Redirect user to Google's login page
    redirect_uri = "https://mlcoach.online/auth/google/callback" # Must match Google Console exactly
    return await oauth.google.authorize_redirect(request, redirect_uri)

@router.get("/auth/google/callback")
async def google_callback(request: Request):
    try:
        # Get the token from Google
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        
        if not user_info:
            raise HTTPException(status_code=400, detail="Failed to get user info")
            
        email = user_info.get('email')
        name = user_info.get('name')
        picture = user_info.get('picture')
        
        # TODO: Check if user exists in your database. 
        # If not, create a new user record.
        
        # Generate your own JWT token for your app
        # token_payload = {"sub": email, "name": name}
        # access_token = create_access_token(data=token_payload)
        
        # Redirect back to your frontend dashboard with the token
        # Example: return RedirectResponse(url=f"https://mlcoach.online/dashboard?token={access_token}")
        
        return {"message": "Login successful", "user": user_info}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
