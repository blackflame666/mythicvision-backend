from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from openai import OpenAI
import tempfile
import os

# 🔑 PASTE YOUR OPENAI API KEY HERE
API_KEY = "YOUR_OPENAI_API_KEY"

# Initialize App and AI
app = FastAPI()
client = OpenAI(api_key=API_KEY)

@app.post("/analyze-match")
async def analyze_match(file: UploadFile = File(...)):
    print(f"📥 Received video: {file.filename}. Processing...")
    
    # 1. Save the uploaded video to a temporary file on the server
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = tmp_file.name

    try:
        # 2. Run your Computer Vision logic (The "Eyes")
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        timeline_data = []
        current_time = 0.0
        last_sampled_time = -10.0 # Sample every 10 seconds

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            if current_time - last_sampled_time >= 10.0:
                last_sampled_time = current_time
                mins = int(current_time // 60)
                secs = int(current_time % 60)
                time_str = f"{mins:02d}:{secs:02d}"
                
                # Crop Minimap (Top-Left 25%)
                h, w = frame.shape[:2]
                minimap = frame[0:int(h*0.25), 0:int(w*0.25)]
                
                # Color Mask (Red/Orange enemies)
                hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
                lower_red = np.array([0, 100, 100])
                upper_red = np.array([10, 255, 255])
                mask = cv2.inRange(hsv, lower_red, upper_red)
                
                # Count Contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                enemy_count = sum(1 for c in contours if cv2.contourArea(c) > 30)
                
                timeline_data.append({"time": time_str, "enemies_visible": enemy_count})
        
        cap.release()

        # 3. Send to OpenAI (The "Brain")
        timeline_string = "\n".join([f"- {d['time']}: {d['enemies_visible']} enemies visible" for d in timeline_data])
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert MLBB coach. Analyze this timeline and give a 3-paragraph report (Laning, Mid, Late game)."},
                {"role": "user", "content": f"Timeline:\n{timeline_string}"}
            ]
        )
        
        ai_report = response.choices[0].message.content
        
        # 4. Send the report back to the website
        return JSONResponse(content={"status": "success", "report": ai_report})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)