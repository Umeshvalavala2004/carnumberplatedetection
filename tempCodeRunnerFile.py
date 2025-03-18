import numpy as np
import cv2
import imutils
import pytesseract
import os

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Read the image file
image_path = 'Car Images/3.jpg'
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    exit()

image = cv2.imread(image_path)

# Resize the image - change width to 500
image = imutils.resize(image, width=500)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter (Noise removal while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Apply Canny edge detection
edged = cv2.Canny(gray, 170, 200)

# Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area, keeping only the largest ones
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCnt = None

# Loop over contours to find the number plate
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:  # Select the contour with 4 corners
        NumberPlateCnt = approx
        x, y, w, h = cv2.boundingRect(c)

        # Crop and save the number plate
        cropped_image = gray[y:y + h, x:x + w]
        cropped_path = 'Cropped Images-Text/7.png'
        os.makedirs(os.path.dirname(cropped_path), exist_ok=True)
        cv2.imwrite(cropped_path, cropped_image)

        break  # Exit after finding the first valid plate

# Draw the detected number plate contour
if NumberPlateCnt is not None:
    cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
    cv2.imshow("Number Plate Detected", image)
else:
    print("No number plate detected.")

# Read and process the cropped number plate image
if os.path.exists(cropped_path):
    cropped_img = cv2.imread(cropped_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding for better OCR results
    _, thresh = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Run Tesseract OCR
    text = pytesseract.image_to_string(thresh, lang='eng')
    print("Detected Number Plate:", text.strip())

    # Show cropped image
    cv2.imshow("Cropped Number Plate", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
