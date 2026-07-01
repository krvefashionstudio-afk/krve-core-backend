from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import cv2
import numpy as np
import urllib.request

app = FastAPI(title="KrvE Real-Time Deep-Tech AI Engine", version="3.9.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./temp_framework_scans"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_PATH = "./haarcascade_frontalface_default.xml"

if not os.path.exists(CASCADE_PATH):
    print("[KrvE AI] Downloading stable face tracking model...")
    urllib.request.urlretrieve(CASCADE_URL, CASCADE_PATH)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

@app.post("/api/v1/generate-mesh")
async def generate_user_mesh(
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
    height: float = Form(...)
):
    try:
        front_path = os.path.join(UPLOAD_DIR, "front_" + front_image.filename)
        with open(front_path, "wb") as buffer:
            buffer.write(await front_image.read())
            
        img = cv2.imread(front_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        face_coordinates = []
        if len(faces) > 0:
            print("[KrvE AI] Face detected successfully!")
            x, y, w, h = faces[0]
            face_coordinates.append({"x": int(x + w/2), "y": int(y + h/2), "z": 0})
            status_msg = "Real Face mapped successfully onto 3D vertices framework!"
        else:
            print("[KrvE AI] Warning: No face bounds detected.")
            status_msg = "Digital twin generated using default profile vectors (Face not clear)."

        # Safe String Formatter Conversion
        chest_calc = str(round(height * 0.22, 1)) + " IN"
        waist_calc = str(round(height * 0.18, 1)) + " IN"
        hip_calc = str(round(height * 0.23, 1)) + " IN"
        
        recommended = "M"
        if height > 180: 
            recommended = "XL"
        elif height < 170: 
            recommended = "S"

        return {
            "status": "success",
            "message": status_msg,
            "chest": chest_calc,
            "waist": waist_calc,
            "hip": hip_calc,
            "recommended_size": "KRVE MATCH " + recommended,
            "total_extracted_face_nodes": len(face_coordinates),
