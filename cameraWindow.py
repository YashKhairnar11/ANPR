import flet as ft
import cv2
import base64
import threading, sqlite3
from modelFactory import ANPRModel, YOLOv11SegmentationModel, YOLOv11DetectionModel  # Import the ML models

class WindowStreamer:
    def __init__(self, cam_name, cam_details):
        self.cam_name = cam_name
        self.cam_details = cam_details
        self.connections = {}  # To manage multiple sources and their states
        self.streaming_windows = {}
        self.type = cam_details['task']

        # Initialize the model based on cam_details
        self.model = self.initialize_model()
        self.data_tables = {}
        self._create_table() # create a table in db
    
    def create_table(self,source_id):
        if self.cam_details['task']=='Anpr':
            """Create ANPR data table"""
            data_table =  ft.DataTable(
                heading_row_color=ft.colors.BLACK12,
                columns=[
                    ft.DataColumn(ft.Text("Time")),
                    ft.DataColumn(ft.Text("ID")),
                    ft.DataColumn(ft.Text("Type")),
                    ft.DataColumn(ft.Text("License Number")),
                ],
                rows=[]
            )
            self.data_tables[source_id]=data_table
            return data_table
          
        else:
            """Create object detection data table"""
            data_table= ft.DataTable(
                    heading_row_color=ft.colors.BLACK12,
                    columns=[
                        ft.DataColumn(ft.Text("Time")),
                        ft.DataColumn(ft.Text("ID")),
                        ft.DataColumn(ft.Text("Type")),
                    ],
                    rows=[]
                )
            self.data_tables[source_id]=data_table
            return data_table
    
    def _create_table(self):
        """Create database table for the camera"""
        with sqlite3.connect('records.db') as conn:
            cursor = conn.cursor()

            if self.cam_details['task'] == 'Anpr':
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.cam_name} (
                    Time TEXT,
                    ID TEXT,
                    Type TEXT,
                    LicenseNumber TEXT 
                )
                """)
                cursor.execute(f"DELETE FROM {self.cam_name}")
            else:
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.cam_name} (
                    Time TEXT,
                    ID TEXT,
                    Type TEXT
                )
                """)
                cursor.execute(f"DELETE FROM {self.cam_name}")
            conn.commit()

    def _fetch_latest_records(self):
        """Fetch the latest 5 records from the database"""
        with sqlite3.connect('records.db') as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * 
                FROM {self.cam_name} 
                ORDER BY ROWID DESC 
                LIMIT 5
            """)
            return cursor.fetchall()
        
    def _update_table(self,source_id):
        """Update the table with latest records"""
        records = self._fetch_latest_records()
        if self.type =='Anpr':
            rows = [
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(record[0]))),  # Time
                        ft.DataCell(ft.Text(record[1])),       # ID
                        ft.DataCell(ft.Text(record[2])),       # Type
                        ft.DataCell(ft.Text(record[3] if len(record) > 3 else '')),  # License Number
                    ]
                )
                for record in records
            ]
        else:
            rows = [
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(record[0]))),  # Time
                        ft.DataCell(ft.Text(record[1])),       # ID
                        ft.DataCell(ft.Text(record[2])),       # Type
                    ]
                )
                for record in records
            ]
        table = self.data_tables[source_id]
        table.rows = rows 
        table.update()

    def initialize_model(self):
        """Initialize the ML model based on camera details."""
        model_used = self.cam_details.get('model_used', '')
        if model_used == 'ANPRModel':
            return ANPRModel()
        elif model_used == 'YOLOv11DetectionModel':
            return YOLOv11DetectionModel()
        elif model_used == 'YOLOv11SegmentationModel':
            return YOLOv11SegmentationModel()
        else:
            return None  # Default to no model if not specified

    def create_streaming_window(self):
        return ft.Container(
            width=720,
            height=480,
            content=ft.Image(
                src="assets/disconnect.png",
                width=50,
                height=50,
                fit=ft.ImageFit.CONTAIN,
            ),
            margin=ft.margin.all(10),
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            bgcolor=ft.colors.BLACK,
            border=ft.border.all(2, ft.colors.OUTLINE),
            expand=True,
        )

    def create_connect_button(self, source_id):
        return ft.ElevatedButton(
            text="Connect",
            style=ft.ButtonStyle(
                color={
                    ft.MaterialState.DEFAULT: ft.colors.WHITE,
                    ft.MaterialState.HOVERED: ft.colors.WHITE,
                },
                bgcolor={
                    ft.MaterialState.DEFAULT: ft.colors.GREEN,
                    ft.MaterialState.HOVERED: ft.colors.GREEN_700,
                },
            ),
            on_click=lambda e: self.toggle_connection(e, source_id)
        )

    def toggle_connection(self, e, source_id):
        if source_id not in self.connections:
            self.connections[source_id] = {
                "is_connected": False,
                "cap": None
            }

        connection = self.connections[source_id]
        connection["is_connected"] = not connection["is_connected"]

        if connection["is_connected"]:
            self.connect(e.control, source_id)
        else:
            self.disconnect(e.control, source_id)

    def connect(self, connect_button, source_id):
        # Update button to "Disconnect"
        connect_button.text = "Disconnect"
        connect_button.style.bgcolor = {
            ft.MaterialState.DEFAULT: ft.colors.RED,
            ft.MaterialState.HOVERED: ft.colors.RED_700,
        }
        connect_button.update()

        source = source_id  # Assign the source (e.g., 0 for webcam or file path)
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Failed to open video source: {source}")
            return

        # Store the video capture object in the connections dictionary
        self.connections[source_id]["cap"] = cap

        # Start a background thread to read frames
        threading.Thread(target=self.read_frames, args=(source_id,), daemon=True).start()

    def disconnect(self, connect_button, source_id):
        # Update button to "Connect"
        connect_button.text = "Connect"
        connect_button.style.bgcolor = {
            ft.MaterialState.DEFAULT: ft.colors.GREEN,
            ft.MaterialState.HOVERED: ft.colors.GREEN_700,
        }
        connect_button.update()

        connection = self.connections.get(source_id, None)
        if connection and connection["cap"]:
            connection["cap"].release()
            connection["cap"] = None

        # Update streaming window to show disconnected state
        window = self.streaming_windows.get(source_id, None)
        if window:
            window.content.src = "assets/disconnect.png"
            window.update()

    def read_frames(self, source_id):
        connection = self.connections[source_id]
        cap = connection["cap"]
        window = self.streaming_windows[source_id]

        frame_num = 0
        while connection['is_connected']:
            frameDict = {}
            ret, frame = cap.read()
            if not ret:
                print("Cannot connect to source!")
                break

            frame_num += 1
            frameDict['frameNum'] = frame_num
            frameDict['frame'] = frame

            try:
                if self.cam_details['model_used'] == 'ANPRModel':
                    frame_dict1 = self.model.det_objects(frameDict)
                    frame_dict2 = self.model.det_plates_ocr(frame_dict1)
                    res_frame = self.model.plot_bounding_boxes(frame, frame_dict2,self.cam_name)
                    # Update UI table
                    self._update_table(source_id)
                elif self.cam_details['model_used'] == 'YOLOv11DetectionModel':
                    frameDict1= self.model.predict(frameDict)
                    res_frame = self.model.plot_bounding_boxes(frame,frameDict1,self.cam_name)
                    self._update_table(source_id)
                else:
                    res_frame = frame
                _, buffer = cv2.imencode(".jpg", res_frame)
                img_str = base64.b64encode(buffer).decode("utf-8")

                window.content.src_base64 = f"{img_str}"
                window.update()

            except Exception as e:
                print(f"Error processing frame: {e}")
                break

    def build(self, source_id, cam_name):
        streaming_window = self.create_streaming_window()
        table = self.create_table(source_id)
        self.streaming_windows[source_id] = streaming_window
        connect_button = self.create_connect_button(source_id)

        controls = [
                    ft.Text(
                        value=cam_name,  # Display the camera name
                        style=ft.TextStyle(size=16, weight="bold"),  # Bold, larger font
                    ),
                    streaming_window,  # Streaming window
                    connect_button,  # Connect button
                    table
        ]

        return ft.Container(
            content=ft.Column(
                controls=controls,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,  # Spacing between elements
                expand=True,  # Allow the column to expand
            ),
            padding=20,
            border_radius=10,
            bgcolor=ft.colors.SURFACE,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=ft.colors.with_opacity(0.3, ft.colors.SHADOW),
            ),
            expand=True,  # Allow the container to expand
        )
