from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
import imutils
import pytesseract
import uuid  # Unique filenames

# Initialize Flask app
app = Flask(__name__)

# Configure directories
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Tesseract OCR (Ensure correct path for Windows users)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_image(image_path, filename):
    """Processes an image to detect a number plate and extract text."""
    image = cv2.imread(image_path)

    if image is None:
        return None, None, "Error: Unable to read image."

    # Resize for consistency
    image = imutils.resize(image, width=500)

    # Convert to grayscale and apply bilateral filter
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge detection
    edged = cv2.Canny(gray, 170, 200)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    number_plate_cnt = None
    cropped_filename = None
    extracted_text = "Number plate not detected."

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:  # Looking for a quadrilateral shape
            number_plate_cnt = approx
            x, y, w, h = cv2.boundingRect(c)
            cropped_img = gray[y:y + h, x:x + w]

            # Save the detected plate with a unique name
            cropped_filename = f"plate_{uuid.uuid4().hex}.png"
            cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], cropped_filename)
            cv2.imwrite(cropped_path, cropped_img)

            # Run OCR
            _, thresh = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            extracted_text = pytesseract.image_to_string(thresh, lang='eng').strip()

            break  # Stop after finding the first valid plate

    # Save processed image with detected plate
    processed_filename = f"processed_{filename}"
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

    if number_plate_cnt is not None:
        cv2.drawContours(image, [number_plate_cnt], -1, (0, 255, 0), 3)
        cv2.imwrite(processed_path, image)

    return processed_filename, cropped_filename, extracted_text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            processed_filename, cropped_filename, extracted_text = process_image(file_path, unique_filename)

            return render_template("result.html", 
                                   result=extracted_text, 
                                   uploaded_image=unique_filename, 
                                   processed_image=processed_filename, 
                                   cropped_image=cropped_filename)

    return render_template("ui.html")  # ❌ Fixed the syntax error

if __name__ == "__main__":
    app.run(port=5000, debug=True)
