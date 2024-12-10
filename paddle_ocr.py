import flet as ft
import cv2
import websockets, aiohttp, time, asyncio
import threading, numpy as np
import easyocr
from ultralytics import YOLO
import logging
import base64, string, csv
from io import BytesIO
from PIL import Image
import sqlite3
from sort.sort import Sort
from threading import Timer
from datetime import datetime
from paddleocr import PaddleOCR
import re

# Mapping dictionaries for character conversion
indian_state_abbreviations = [
    "AP",  # Andhra Pradesh
    "AR",  # Arunachal Pradesh
    "AS",  # Assam
    "BR",  # Bihar
    "CG",  # Chhattisgarh
    "GA",  # Goa
    "GJ",  # Gujarat
    "HR",  # Haryana
    "HP",  # Himachal Pradesh
    "JH",  # Jharkhand
    "JK",  # Jammu and Kashmir
    "KA",  # Karnataka
    "KL",  # Kerala
    "LD",  # Lakshadweep
    "MP",  # Madhya Pradesh
    "MH",  # Maharashtra
    "MN",  # Manipur
    "ML",  # Meghalaya
    "MZ",  # Mizoram
    "NL",  # Nagaland
    "OD",  # Odisha
    "PB",  # Punjab
    "RJ",  # Rajasthan
    "SK",  # Sikkim
    "TN",  # Tamil Nadu
    "TS",  # Telangana
    "TR",  # Tripura
    "UP",  # Uttar Pradesh
    "UK",  # Uttarakhand
    "WB",  # West Bengal
    "AN",  # Andaman and Nicobar Islands
    "CH",  # Chandigarh
    "DN",  # Dadra and Nagar Haveli and Daman and Diu
    "DL",  # Delhi
    "PY"   # Puducherry
]

char_to_int_misreads = {
    'O': 0,  # O looks like 0
    'D': 0,  # D can resemble 0
    'Q': 0,  # Q can resemble 0
    'I': 1,  # I looks like 1
    'L': 1,  # L looks like 1
    'Z': 2,  # Z looks like 2
    'E': 3,  # E looks like 3
    'A': 4,  # A looks like 4
    'S': 5,  # S looks like 5
    'G': 6,  # G looks like 6
    'T': 7,  # T looks like 7
    'B': 8,  # B looks like 8
    'g': 9,  # g looks like 9
    'q': 9   # q looks like 9
}

############################################
class App:
    def __init__(self,source):

        self.source = source
        self.cap = cv2.VideoCapture('output_video2.mp4')
        self.is_connected = False
        
        # load detection and ocr models
        self.objectModel = YOLO("yolo11n.pt")
        self.plateModel = YOLO('license_plate_detector.pt')
        #self.ocr = easyocr.Reader(['en'],gpu=True)
        self.ocr = PaddleOCR(lang='en',det=False, cls=False)
        self.classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 
                        'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
                        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
                        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
                        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
                        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
                        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
                        70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
        self.vehicle_class = {2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}
        self.tracker = Sort()

        #state varialbles
        self.wiper_active = None
        self.left_active = False
        self.right_active = False
        self.up_active = False
        self.down_active = False
        self.zoomIn = False
        self.zoom_label = ft.Text("Zoom Level: 0%")

        ###### UI initialization #####
        # Initializing the title bar of the app
        self.titlebar = ft.AppBar(
            leading=ft.Icon(ft.icons.CAMERA_ALT),
            title=ft.Text("Automatic Number Plate Recognition"),
            center_title=True,
            bgcolor=ft.colors.BLUE,
        )
        
        # The streaming window
        self.streamingWindow = ft.Container(
            width=720,
            height=480,
            content=ft.Image(
                src="assets/disconnect.png",  # Fixed path separator
                width=100,
                height=100,
                fit=ft.ImageFit.CONTAIN,  # Added image fitting
            ),
            alignment=ft.alignment.center,
            border=ft.border.all(2, ft.colors.BLACK),
            margin=ft.margin.all(10),  # Added margin
        )

        # Button for connecting to the camera
        self.connectButton = ft.ElevatedButton(  # Removed extra Container
            text="Connect to Camera",
            width=200,  # Added width
            bgcolor="Green",
            color='black',
            on_click=self.connect_or_disconnect_camera
        )

        # Camera Controls
        self.controls = ft.Container(
            content=ft.Column(
                controls=[
                    # Top row with Up button
                    ft.Row(
                        [ft.Container(width=60), 
                         ft.IconButton(icon=ft.icons.KEYBOARD_ARROW_UP, 
                                     icon_size=40,
                                     width=60,
                                     height=60,
                                     on_click=self.toggle_up),
                         ft.Container(width=60)],
                        alignment=ft.MainAxisAlignment.CENTER
                    ),
                    # Middle row with Left, Wiper, Right buttons
                    ft.Row(
                        [
                            ft.IconButton(icon=ft.icons.KEYBOARD_ARROW_LEFT,
                                        icon_size=40,
                                        width=60,
                                        height=60,
                                        on_click=self.toggle_left),
                            ft.IconButton(icon=ft.icons.DRY_CLEANING,
                                        icon_size=40,
                                        width=60,
                                        height=60,
                                        on_click=self.toggle_wiper),
                            ft.IconButton(icon=ft.icons.KEYBOARD_ARROW_RIGHT,
                                        icon_size=40,
                                        width=60,
                                        height=60,
                                        on_click=self.toggle_right),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER
                    ),
                    # Bottom row with Down button
                    ft.Row(
                        [ft.Container(width=60),
                         ft.IconButton(icon=ft.icons.KEYBOARD_ARROW_DOWN,
                                     icon_size=40,
                                     width=60,
                                     height=60,
                                     on_click=self.toggle_down),
                         ft.Container(width=60)],
                        alignment=ft.MainAxisAlignment.CENTER
                    ),  
                    # Zoom control with slide
                     ft.Column(
                        controls=[
                            ft.Text("Zoom Control", size=16, weight=ft.FontWeight.BOLD),
                            ft.Row(
                                controls=[
                                    ft.Icon(ft.icons.ZOOM_OUT, size=20),
                                    ft.Slider(
                                        min=0,
                                        max=100,
                                        divisions=10,
                                        label="{value}%",
                                        width=200,
                                        active_color=ft.colors.BLUE,
                                        # on_change=self.
                                    ),
                                    ft.Icon(ft.icons.ZOOM_IN, size=20),
                                ],
                                alignment=ft.MainAxisAlignment.CENTER,
                                spacing=10,
                            ),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=10,
                    )
                ],
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER
            ),
            padding=ft.padding.all(20),
        )

        #vehicle record table
        self.record_table=ft.DataTable(
            heading_row_color=ft.colors.BLACK12,
            columns=[
                ft.DataColumn(ft.Text("Time")),
                ft.DataColumn(ft.Text("ID")),
                ft.DataColumn(ft.Text("Type")),
                ft.DataColumn(ft.Text("License Number")),
            ],
            rows = []
        )

    def build(self):
        return ft.Container(  # Wrapped in Container for better layout control
            content=ft.Row(
                controls=[
                    self.controls,
                    ft.Column(
                        controls=[
                            self.streamingWindow,
                            self.connectButton,
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=20,  # Added spacing
                    ),
                    self.record_table
                ],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            margin=ft.margin.all(5),
        )
    
    def did_mount(self):
        conn = sqlite3.connect('anpr.db',check_same_thread=False)
        cur = conn.cursor()
        # Clean the table by deleting all rows
        cur.execute("DROP TABLE vehiclerecords")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vehiclerecords (
            Time TEXT,
            ID TEXT,
            Type TEXT,
            LicenseNumber TEXT )""")
        conn.commit()
        conn.close()
    
    ######## database funtions ############
    # Insert Detection Data into Database
    def insert_detection(self,tracking_id, vehicle_type, license_number):
        conn = sqlite3.connect('anpr.db',check_same_thread=False)
        cur = conn.cursor()
        try:
             # Check if the record already exists based on the tracking_id
            cur.execute("SELECT 1 FROM vehiclerecords WHERE ID = ? OR LicenseNumber = ?", (tracking_id,license_number))
            existing_record = cur.fetchone()
            if existing_record:
                print(f"TrackingID {tracking_id} already exists.")
            else:
                detection_time = datetime.now().strftime("%H:%M:%S")
                cur.execute(
                    "INSERT INTO vehiclerecords (Time, ID , Type , LicenseNumber) VALUES (?, ?, ?, ?)",
                    (detection_time, tracking_id, vehicle_type, license_number),
                )
                conn.commit()
        except sqlite3.IntegrityError:
            print(f"TrackingID {tracking_id} already exists.")
        conn.close()
        self.update_table()

    # Fetch first 5 records from the database 
    def fetch_latest_records(self):
         conn = sqlite3.connect('anpr.db',check_same_thread=False)
         cur = conn.cursor()
         cur.execute( """
            SELECT Time, ID , Type , LicenseNumber 
            FROM vehiclerecords 
            ORDER BY ROWID DESC 
            LIMIT 5
            """)
         data = cur.fetchall()
         return data
    
    def create_table_rows(self):
        # Fetch the latest 5 records from the database
        records = self.fetch_latest_records()

        # Convert database rows into DataRow objects
        return [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(str(record[0]))),  # DetectionTime
                    ft.DataCell(ft.Text(record[1])),       # Track ID
                    ft.DataCell(ft.Text(record[2])),       # vehcle type
                    ft.DataCell(ft.Text(record[3])),       # License Number
                ]
            )
            for record in records
        ]

    def update_table(self):
        # Fetch the latest rows and update the table
        self.record_table.rows = self.create_table_rows()
        self.record_table.update()

    ######### Utility functions ############
    async def get_session_cookie(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                cookies = session.cookie_jar.filter_cookies(url)
                session_cookie = cookies.get('PHPSESSID')
                return session_cookie.value if session_cookie else ""

    async def connect_or_disconnect_camera(self,e):
        if self.is_connected:
            self.is_connected = False
            self.streamingWindow.content.src = "assets/disconnect.png"
            self.connectButton.text = "Connect"
            self.connectButton.bgcolor = "Green"
            self.connectButton.color = "Black"
            self.streamingWindow.update()
            await self.disconnect()
        else:
            await self.connect()



    async def connect(self):
        if self.source == 0:
            self.is_connected = True
            self.connectButton.text = "Disconnect"
            self.connectButton.bgcolor = "Red"
            self.connectButton.color = "White"
            self.connectButton.update()
            self.start_video_streaming()
            print("Connected to the webcam")
        else:
            url = 'ws://192.168.1.111/cgi-bin/event-websock/streaming.cgi'
            base_url = 'http://192.168.1.111'

            try:
                session_cookie = await self.get_session_cookie(base_url)
                headers = {
                    'Origin': base_url,
                    'Cookie': f'PHPSESSID={session_cookie}'
                }

                # Create a single client session that can be reused
                self.session = aiohttp.ClientSession()

                # Establish WebSocket connection with more robust error handling
                self.websocket = await self.session.ws_connect(
                    url, 
                    headers=headers,
                    heartbeat=30,  # Add a heartbeat to keep connection alive
                    timeout=10     # Set a connection timeout
                )

                print(f"Successfully connected to {url}")
                self.is_connected = True
                self.connectButton.text = "Disconnect"
                self.connectButton.bgcolor = "Red"
                self.connectButton.color = "White"
                self.connectButton.update()
                self.start_video_streaming()

            except aiohttp.ClientConnectorError as e:
                print(f"Connection error: {e}")
                self.is_connected = False
            except Exception as e:
                print(f"Failed to connect: {type(e).__name__}: {e}")
                self.is_connected = False

    
    
    async def disconnect(self):
     if hasattr(self, 'websocket') and self.websocket:
        await self.websocket.close()
     if hasattr(self, 'session') and self.session:
        await self.session.close()
        self.is_connected = False
        self.connectButton.text = "Connect"
        self.connectButton.bgcolor = "Green"
        self.connectButton.color = "Black"
        self.connectButton.update()
    

    def start_video_streaming(self):
        threading.Thread(target=self.read_frames, daemon=True).start()
    

    def read_frames(self):
        frameNum = 0
        while self.is_connected:
            frameDict = {}
        
            ret, frame = self.cap.read()
            if not ret:
                print("Cannot connect to source !")
                break

            frameNum+=1
            frameDict['frameNum'] = frameNum
            frameDict['frame'] = frame
            
            frame_dict1 = self.det_objects(frameDict)
            frame_dict2 = self.det_plates_ocr(frame_dict1)
            res_frame = self.plot_bounding_boxes(frame,frame_dict2)

            # Encode the frame in JPEG format and convert to base64
            _, buffer = cv2.imencode(".jpg", res_frame)
            img_str = base64.b64encode(buffer).decode("utf-8")

            # Update the streaming window with the live frame
            self.streamingWindow.content.width = 720
            self.streamingWindow.content.height = 480
            self.streamingWindow.content.src_base64 = f"{img_str}"
            self.streamingWindow.update()


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
                        self.insert_detection(objectID, objectType, plate_text)
                   
        # Return the frame with bounding boxes
        return frame

    ######### cam control functions #################
    async def send_message(self, message):
        try:
            if hasattr(self, 'websocket') and not self.websocket.closed:
                await self.websocket.send_str(message)
                print(f"Sent message: {message}")
            else:
                print("WebSocket is not connected or has been closed")
        except Exception as e:
         print(f"Failed to send message: {type(e).__name__}: {e}")
         # Attempt to reconnect
         await self.connect()

    async def toggle_wiper(self,e):
        self.wiper_active = not self.wiper_active
        message = "type=ptz&aux_on=2&user=admin&host=192.168.1.135" if self.wiper_active else "type=ptz&aux_on=0&user=admin&host=192.168.1.135"
        await self.send_message(message)
        
    async def toggle_left(self,e):
        self.left_active = not self.left_active
        message = "type=ptz&move=left&pspd=30&user=admin&host=192.168.1.135" if self.left_active else "type=ptz&move=stop&user=admin&host=192.168.1.135"
        await self.send_message(message)
        autofocus_msg = "type=ptz&focus=pushaf&user=admin&host=192.168.1.135"
        await self.send_message(autofocus_msg)

    async def toggle_right(self,e):
        self.right_active = not self.right_active
        message = "type=ptz&move=right&pspd=30&user=admin&host=192.168.1.135" if self.right_active else "type=ptz&move=stop&user=admin&host=192.168.1.135"
        await self.send_message(message)
        autofocus_msg = "type=ptz&focus=pushaf&user=admin&host=192.168.1.135"
        await self.send_message(autofocus_msg)
        
    async def toggle_down(self,e):
        self.down_active = not self.down_active
        message = "type=ptz&move=down&pspd=30&user=admin&host=192.168.1.135" if self.down_active else "type=ptz&move=stop&user=admin&host=192.168.1.135"
        await self.send_message(message)
        autofocus_msg = "type=ptz&focus=pushaf&user=admin&host=192.168.1.135"
        await self.send_message(autofocus_msg)

    async def toggle_up(self,e):
        self.up_active = not self.up_active
        message = "type=ptz&move=up&pspd=30&user=admin&host=192.168.1.135" if self.up_active else "type=ptz&move=stop&user=admin&host=192.168.1.135"
        await self.send_message(message)
        autofocus_msg = "type=ptz&focus=pushaf&user=admin&host=192.168.1.135"
        await self.send_message(autofocus_msg)
   
    async def zoom_in(self,e):
        self.zoomIn = not self.zoomIn
        message = "type=ptz&zoom=tele&zspd=5&user=admin&host=192.168.1.135" if self.zoomIn else "type=ptz&zoom=stop&user=admin&host=192.168.1.135"
        await self.send_message(message)
        autofocus_msg = "type=ptz&focus=pushaf&user=admin&host=192.168.1.135"
        await self.send_message(autofocus_msg)

    async def zoom_out(self,e):
        message = "type=ptz&zoom=wide&zspd=30&user=admin&host=192.168.1.135"
        await self.send_message(message)
        stop_msg =   "type=ptz&zoom=stop&zspd=30&user=admin&host=192.168.1.135"
        await self.send_message(stop_msg)
        autofocus_msg = "type=ptz&focus=pushaf&user=admin&host=192.168.1.135"
        await self.send_message(autofocus_msg)
    
    def update_zoom(self, value):
         asyncio.run(self.async_update_zoom(value))

    async def async_update_zoom(self, value):
        zoom_value = int(value)  # Get the value of the slider
        self.zoom_label.configure(text=f"Zoom Level: {zoom_value}%")  # Update the label with the zoom percentage
        message = f"type=ptz&position=set&zoom_pos={zoom_value}&user=admin&host=192.168.1.135"
        await self.send_message(message)
        autofocus_msg = "type=ptz&focus=pushaf&user=admin&host=192.168.1.135"
        await self.send_message(autofocus_msg)


    async def stop(self,e):
        await self.send_message("type=ptz&move=stop&user=admin&host=192.168.1.135")


############### main function #################################
def main(page: ft.Page):
    source = 0
    # Create an instance of the App
    inst = App(source)
    inst.did_mount()

    # Configure the page
    page.title = "ANPR System"
    page.appbar = inst.titlebar
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.theme_mode = "LIGHT"

    # Add the content to the page
    page.add(inst.build())
    page.update()

ft.app(target=main)