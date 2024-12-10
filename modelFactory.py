from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
import cv2
from sort.sort import Sort
import base64
import numpy as np

class BaseModel:
    """A base class for all models. Define the interface here."""
    def predict(self, frame):
        raise NotImplementedError("Predict method should be implemented by the specific model subclass!!!")

############ Object Detection #################################
class YOLOv11DetectionModel(BaseModel):
    def __init__(self, model_path='yolo11n.pt'):
        self.model = YOLO(model_path)

    def predict(self, frame):
        # Perform detection
        results = self.model(frame, task="detect")  # Detection task
        detections = results[0]  # Extract detection results
        
        # Plot bounding boxes on the frame
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            label = f"Class {int(class_id)}: {score:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box
            
            # Add label text
            cv2.putText(
                frame, 
                label, 
                (int(x1), int(y1) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA
            )
        
        # Return the annotated frame
        return frame

########### Segmentation Model #################################
class YOLOv11SegmentationModel(BaseModel):
    def __init__(self, model_path='yolov11n-seg.pt'):
        self.model = YOLO(model_path)

    def predict(self, frame):
        results = self.model(frame, task="segment")  # Segmentation task
        return results[0]

######## ANPR for number plate detection ######################
class ANPRModel(BaseModel):
    def __init__(self):
        self.objectModel = YOLO("yolo11n.pt")
        self.plateModel = YOLO('license_plate_detector.pt')
        self.ocr = PaddleOCR(lang='en',det=False, cls=False)
        self.tracker = Sort()
        self.classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 
                        'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
                        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
                        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
                        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
                        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
                        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
                        70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
        self.vehicle_class = {2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}
           

    def det_objects(self,frameDict):
            frame = frameDict['frame']
            # Perform inference
            results = self.objectModel.predict(frame)[0]  #,classes=list(self.vehicle_class.keys())
            # Store detected vehicle data
            detected_objects = []
            to_be_tracked_objects = []
            for object in results.boxes.data.tolist():
                x1_o, y1_o, x2_o, y2_o, score, class_obj = object
                # Extract relevant information about each detected vehicle
                object_data = {
                    'obj_bbox': [int(x1_o), int(y1_o), int(x2_o), int(y2_o)],
                    'type': self.classes[int(class_obj)],
                    'confidence': score,
                    'plates':[],
                    'trackID':""
                    }
                detected_objects.append(object_data)
                # Create tracking box
                to_be_tracked_objects.append([int(x1_o), int(y1_o), int(x2_o), int(y2_o), score])

            # Add the detection results to frameDict
            frameDict['detected_objects'] = detected_objects

            if to_be_tracked_objects:
                track_ids = self.tracker.update(np.asarray(to_be_tracked_objects))
                # matching tracking ids with detections
                for obj in frameDict['detected_objects']:
                    bbox = obj['obj_bbox']
                    for track in track_ids:
                        if self.boxes_match(bbox,track[:4].tolist()):
                            obj['trackID'] = int(track[4])
                            break

            return frameDict
    
    @staticmethod
    def boxes_match(bbox1, bbox2, threshold=5):
        """Compare two bounding boxes for approximate matching."""
        return all(abs(float(bbox1[i]) - float(bbox2[i])) <= threshold for i in range(4))

    def det_plates_ocr(self,frameDict):
            frame = frameDict['frame']
            # Process each detected vehicle
            for obj in frameDict.get('detected_objects', []):
                coordinates = obj['obj_bbox']
                type_of_object = obj['type']
                if type_of_object in self.vehicle_class.values():
                    # Crop the vehicle region from the frame
                    x_min, y_min, x_max, y_max = coordinates
                    vehicle_crop = frame[y_min:y_max, x_min:x_max]
                    # Perform license plate detection on the cropped image
                    plate_results = self.plateModel.predict(vehicle_crop)[0]
                    # Add detected plates to the respective vehicle data
                    for plate in plate_results.boxes.data.tolist():
                        x1, y1, x2, y2, score, plate_id = plate
                        plate_roi = vehicle_crop[int(y1):int(y2),int(x1):int(x2)]

                       # Perform OCR on the region of interest (plate_roi)
                        ocr_result = self.ocr.ocr(plate_roi)

                        # Extract the text from the OCR result, skipping None values
                        lines = [res[1][0] for i in ocr_result if i is not None for res in i if res[1] is not None and res[1][0] is not None]
                        # Concatenate all the lines into a single string
                        final_text = "".join(lines)

                        print("Final text:", final_text)

                        if final_text!='':
                            final_text = self.format_license(final_text) # format the text as per rules
                            
                            
                        plate_data = {
                            'plate_bbox': [ int(x1), int(y1), int(x2), int(y2)],
                            'text' : final_text
                        }
                        obj['plates'].append(plate_data)
                else:
                    continue
            return frameDict


    def format_license(self,text):
        """
        Format the license plate text by converting characters using the mapping dictionaries.
        Args:
            text (str): License plate text.
        Returns:
            str: Formatted license plate text.
        """
        license_plate_ = ''
        # Remove invalid characters
        ocr_text = re.sub(r'[^A-Za-z0-9]', '', text)
        patterns = [
                    r'^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{1,4}$',  # Standard plates
                    # r'^[A-Z]{2}-TEMP-\d{1,5}$',                      # Temporary plates
                    # r'^CD\s\d{1,3}\s\d{1,4}$',                       # Diplomatic plates
                    # r'^[A-Z]{1,2}\s\d{1,2}\s[A-Z]{1}\d{1,4}$',       # Military plates
                    # r'^[A-Z]{2}\s\d{1,2}\sG\s\d{1,4}$',              # Government plates
                    # r'^[A-Z]{2}\s\d{1,2}\sEV\s\d{1,4}$',             # Electric vehicles
                    # r'^[A-Z]{2}\s\d{1,2}\sV\s[A-Z]{1,2}\d{1,2}$',    # Vintage plates
                    # r'^[A-Z]{2}\sBH\s\d{2}\s\d{1,4}$',               # Bharat series
                    # r'^[A-Z]{2}\s\d{1,2}\sTR\s\d{1,4}$',             # Test vehicles
             ]
        for pattern in patterns:
            if re.match(pattern, ocr_text):
                state = ocr_text[:2]
                district = ocr_text[2:4]
                alphacode = ocr_text[4:6]
                digits = ocr_text[6:]
                
                # # update with corrections
                # if state not in indian_state_abbreviations:
                    
                
                license_plate_ = f"{state}-{district}-{alphacode}-{digits}"
        return license_plate_


    def plot_bounding_boxes(self,frame,frameDict):
        """
        Function to plot bounding boxes for detected vehicles and plates on the frame.
        Args:
            frameDict (dict): Dictionary containing frame and detected vehicles and plates data.
        Returns:
            frame (numpy.ndarray): The frame with bounding boxes drawn.
        """
        # Iterate over each detected vehicle
        for obj in frameDict.get('detected_objects', []):
            # Draw bounding box around the vehicle
            objectType = obj['type']
            objectID = obj['trackID']
            object_coords = obj['obj_bbox']
            xv1, yv1, xv2, yv2 = object_coords
            frame = cv2.rectangle(frame, (xv1,yv1), (xv2,yv2), (0, 255, 0), 2)  # Green box for vehicle
            frame = cv2.putText(frame, f"{objectType}", (xv1, yv1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
            if objectType in self.vehicle_class.values():
                # Check if plates were detected for the vehicle
                for plate in obj.get('plates', []):
                    # Draw bounding box around the plate
                    plate_coords = plate['plate_bbox']
                    x_min, y_min, x_max, y_max = plate_coords

                    frame = cv2.rectangle(frame, (x_min+xv1, y_min+yv1), (x_max+xv1, y_max+yv1), (255,165,0), 2)  # Red box for plate

                    # Optionally, add plate text as a label
                    plate_text = plate['text']

                    if plate_text != '':
                        frame = cv2.putText(frame, plate_text, (x_min+xv1, y_min+yv1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,165,0), 2, cv2.LINE_AA)
                         #update the db if the record doesn't exists already 
                        #self.insert_detection(objectID, objectType, plate_text)
                   
        # Return the frame with bounding boxes
        return frame