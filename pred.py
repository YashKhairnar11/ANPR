from ultralytics import YOLO
import cv2
import numpy as np
import easyocr

# Load the model
model = YOLO("train3/weights/best.pt")  # Load a custom model
reader = easyocr.Reader(['en'])  # Load EasyOCR model 
state_codes = [
    "AN", "AP", "AR", "AS", "BR", "CH", "DN", "DD", "DL", "GA",
    "GJ", "HR", "HP", "JK", "KA", "KL", "LD", "MP", "MH", "MN",
    "ML", "MZ", "NL", "OR", "PY", "PN", "RJ", "SK", "TN", "TR",
    "UP", "WB"
]

# Read the image
image = cv2.imread("car2.jpeg")

# Predict with the model
results = model(image)  # Predict on an image

# Get the bounding boxes
boxes = results[0].boxes

# Draw the bounding boxes on the image and extract cropped plates
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bbox coordinates
    # Draw the rectangle with a specific color and thickness
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    
    # Crop the number plate region
    number_plate = image[y1:y2, x1:x2]
    #res = cv2.resize(number_plate,(525,175))
    number_plate_gray = cv2.cvtColor(number_plate, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(number_plate_gray, 11, 11, 17)
    #_, th1 = cv2.threshold(number_plate_gray,75,255,cv2.THRESH_BINARY)
    # number_plate = cv2.resize(number_plate,(480,320))
    cv2.imshow("number plate",number_plate )
    # Apply EasyOCR on the cropped region
    ocr_result = reader.readtext(number_plate)
    
    # Check if OCR detected any text
    if ocr_result:
        text = ocr_result[0][1]  # Extract the recognized text
        print("Detected Text : ",text)
        #correct the text if mistaken
        state = text[:2]
        print(state)
        # Put the OCR result text on the original image
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, color=(0, 0, 255), thickness=2)

# Display the image with bounding boxes and OCR text
cv2.imshow("Image with Bounding Boxes and OCR Text", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
