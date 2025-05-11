import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf

# Placeholder function for autocomplete (Replace with actual implementation)
def get_autocomplete_suggestions(text):
    if len(text) == 0:
        return []
    # Example: Returns 3 random suggestions based on the last letter
    return [text + "ing", text + "ed", text + "s"]

# Enable OpenCV GPU acceleration
cv2.ocl.setUseOpenCL(True)

# Initialize camera and modules
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Set height
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Constants
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 
          "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "1", "2", 
          "3", "4", "5", "6", "7", "8", "9", "0"]

# Buffer to store detected letters
letter_buffer = []
max_buffer_length = 10  # Store last 10 letters

# UI Settings
window_width = 800
window_height = 600

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.resize(img, (window_width, window_height - 100))  # Resize camera feed
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    suggestions = []  # Reset suggestions for each frame

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure crop coordinates are within bounds
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        # Resize and pad to imgSize
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

        # Get prediction
        prediction, index = classifier.getPrediction(imgWhite)
        detected_letter = labels[index]

        # Store detected letters in buffer
        letter_buffer.append(detected_letter)
        if len(letter_buffer) > max_buffer_length:
            letter_buffer.pop(0)

        # Convert buffer to string
        current_text = "".join(letter_buffer)
        print(f"Detected Text: {current_text}")  # Debugging print

        # Get word suggestions
        suggestions = get_autocomplete_suggestions(current_text)
        print(f"Suggestions: {suggestions}")  # Debugging print

        # Ensure at least one suggestion is available
        if not suggestions:
            suggestions = ["No suggestions"]

        num_suggestions = min(len(suggestions), 3)  # Limit to 3 suggestions

        # Display detected letter
        cv2.putText(imgOutput, f"Detected: {detected_letter}", (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

    else:
        num_suggestions = 1  # Default to 1 if no hands are detected
        suggestions = ["No hands detected"]

    # Draw Auto-Suggestion UI
    box_width = (window_width - 40) // num_suggestions  
    suggestion_box_y = window_height - 80  

    for i in range(num_suggestions):
        box_x = 20 + i * box_width  # Position each box
        cv2.rectangle(imgOutput, (box_x, suggestion_box_y),
                      (box_x + box_width - 10, window_height - 20), (0, 255, 0), -1)
        word = suggestions[i] if i < len(suggestions) else "..."
        cv2.putText(imgOutput, word, (box_x + 10, window_height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show final output
    cv2.imshow("GestureGuru - Sign Recognition", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
