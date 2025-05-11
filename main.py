from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
import io
from PIL import Image

app = FastAPI()

# Allow CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to only allow your Flutter app in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize once
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "1", "2", "3", "4",
    "5", "6", "7", "8", "9", "0"
]
offset = 20
imgSize = 300
letter_buffer = []

def get_autocomplete_suggestions(text):
    if len(text) == 0:
        return []
    return [text + "ing", text + "ed", text + "s"]


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    hands, _ = detector.findHands(img)

    if not hands:
        return {"detected": None, "suggestions": ["No hands detected"]}

    hand = hands[0]
    x, y, w, h = hand['bbox']

    y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
    x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
    imgCrop = img[y1:y2, x1:x2]

    if imgCrop.size == 0:
        return {"detected": None, "suggestions": ["Invalid crop"]}

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    aspectRatio = h / w

    if aspectRatio > 1:
        scale = imgSize / h
        wCal = math.ceil(scale * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = (imgSize - wCal) // 2
        imgWhite[:, wGap:wCal + wGap] = imgResize
    else:
        scale = imgSize / w
        hCal = math.ceil(scale * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = (imgSize - hCal) // 2
        imgWhite[hGap:hCal + hGap, :] = imgResize

    prediction, index = classifier.getPrediction(imgWhite)
    detected_letter = labels[index]

    letter_buffer.append(detected_letter)
    if len(letter_buffer) > 10:
        letter_buffer.pop(0)

    current_text = "".join(letter_buffer)
    suggestions = get_autocomplete_suggestions(current_text)

    return {
        "detected": detected_letter,
        "buffer": current_text,
        "suggestions": suggestions[:3] if suggestions else ["No suggestions"]
    }
