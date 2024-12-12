import flet as ft
import aiohttp
import threading
import cv2
import base64
import sqlite3
import datetime
from modelFactory import ANPRModel, YOLOv11SegmentationModel, YOLOv11DetectionModel

class CameraWindow:
    def __init__(self, cam_name, cam_details):
        self.camDetails = cam_details
        self.cam_name = cam_name
        self.type = cam_details['type']
        self.source = cam_details['source']
        self.cap = None
        self.thread = None
        self.model = self._initialize_model()
        self.is_connected = False
        
        # Ensure thread-safe database connection
        self.db_lock = threading.Lock()
        self._create_table()       
        # Streaming window
        self.streaming_window = ft.Container(
            width=720,
            height=480,
            content=ft.Image(
                src="assets/disconnect.png",
                width=50,
                height=50,
                fit=ft.ImageFit.CONTAIN,
            ),
            border=ft.border.all(2, ft.colors.OUTLINE),
            margin=ft.margin.all(10),
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            expand=True,
        )
        
        # Connect button
        self.connect_button = ft.ElevatedButton(
            text=f"Connect {cam_name}",
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
            on_click=self._toggle_connection
        )
        
        # PTZ controls
        self.ptz_controls = self._create_ptz_controls() if self.type in ['ptz', 'ptz_fixed'] else None
        
        # Record tables
        self.anpr_table = self._create_anpr_table() if self.camDetails['task'] == 'Anpr' else None
        self.obj_table = self._create_obj_table() if self.camDetails['task'] == 'Detection' else None

    def _create_table(self):
        """Create database table for the camera"""
        with sqlite3.connect('records.db') as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.cam_name}")

            if self.camDetails['task'] == 'Anpr':
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.cam_name} (
                    Time TEXT,
                    ID TEXT,
                    Type TEXT,
                    LicenseNumber TEXT 
                )
                """)
            else:
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.cam_name} (
                    Time TEXT,
                    ID TEXT,
                    Type TEXT
                )
                """)
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


    def _update_table(self):
        """Update the table with latest records"""
        records = self._fetch_latest_records()
        if self.type=='Anpr':
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
        if self.anpr_table:
            self.anpr_table.rows = rows
            self.anpr_table.update()
        elif self.obj_table:
            self.obj_table.rows = rows
            self.obj_table.update()

    def _initialize_model(self):
        """Initialize the appropriate model based on configuration"""
        model_map = {
            'ANPRModel': ANPRModel,
            'YOLOv11DetectionModel': YOLOv11DetectionModel,
            'YOLOv11SegmentationModel': YOLOv11SegmentationModel
        }
        return model_map.get(self.camDetails['model_used'], lambda: None)()

    def _create_ptz_controls(self):
        """Create PTZ control buttons"""
        return ft.Column([
            ft.Row([
                ft.IconButton(ft.icons.ARROW_UPWARD, icon_color=ft.colors.BLUE, disabled=self.type == 'ptz_fixed'),
                ft.IconButton(ft.icons.ARROW_DOWNWARD, icon_color=ft.colors.BLUE, disabled=self.type == 'ptz_fixed'),
                ft.IconButton(ft.icons.ARROW_BACK, icon_color=ft.colors.BLUE, disabled=self.type == 'ptz_fixed'),
                ft.IconButton(ft.icons.ARROW_FORWARD, icon_color=ft.colors.BLUE, disabled=self.type == 'ptz_fixed'),
                ft.IconButton(ft.icons.VOLUME_UP, icon_color=ft.colors.RED),
                ft.Slider(min=0, max=100, divisions=10, label="{value}%", expand=True),
            ], alignment=ft.MainAxisAlignment.CENTER),
        ])

    def _create_anpr_table(self):
        """Create ANPR data table"""
        return ft.DataTable(
            heading_row_color=ft.colors.BLACK12,
            columns=[
                ft.DataColumn(ft.Text("Time")),
                ft.DataColumn(ft.Text("ID")),
                ft.DataColumn(ft.Text("Type")),
                ft.DataColumn(ft.Text("License Number")),
            ],
            rows=[]
        )

    def _create_obj_table(self):
        """Create object detection data table"""
        return ft.DataTable(
            heading_row_color=ft.colors.BLACK12,
            columns=[
                ft.DataColumn(ft.Text("Time")),
                ft.DataColumn(ft.Text("ID")),
                ft.DataColumn(ft.Text("Type")),
            ],
            rows=[]
        )

    async def _get_session_cookie(self, url):
        """Retrieve session cookie for PTZ cameras"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                cookies = session.cookie_jar.filter_cookies(url)
                session_cookie = cookies.get('PHPSESSID')
                return session_cookie.value if session_cookie else ""

    async def _connect(self, e):
        """Connect to the camera source"""
        self.connect_button.text = f"Disconnect {self.cam_name}"
        self.connect_button.style.bgcolor = {
            ft.MaterialState.DEFAULT: ft.colors.RED,
            ft.MaterialState.HOVERED: ft.colors.RED_700, 
        }
        self.connect_button.update()

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"Failed to open video source: {self.source}")
            return

        # PTZ camera connection
        if self.type in ['ptz', 'ptz_fixed']:
            base_url = self.camDetails['base_url']
            url = self.camDetails['url']
            try:
                session_cookie = await self._get_session_cookie(base_url)
                headers = {
                    'Origin': base_url,
                    'Cookie': f'PHPSESSID={session_cookie}'
                }

                self.session = aiohttp.ClientSession()
                self.websocket = await self.session.ws_connect(
                    url, 
                    headers=headers,
                    heartbeat=30,
                    timeout=10
                )

                self.is_connected = True
            except Exception as e:
                print(f"Connection error: {e}")
                self.is_connected = False

        # Start video streaming
        self._start_video_stream()

    def _start_video_stream(self):
        """Start video streaming in a separate thread"""
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()

    def _read_frames(self):
        """Continuously read and process video frames"""
        frame_num = 0
        while self.is_connected:
            frameDict = {}
            ret, frame = self.cap.read()
            if not ret:
                print("Cannot connect to source!")
                break

            frame_num += 1
            frameDict['frameNum'] = frame_num
            frameDict['frame'] = frame

            try:
                if self.camDetails['model_used'] == 'ANPRModel':
                    frame_dict1 = self.model.det_objects(frameDict)
                    frame_dict2 = self.model.det_plates_ocr(frame_dict1)
                    res_frame = self.model.plot_bounding_boxes(frame, frame_dict2,self.cam_name,self.db_lock)
                    # Update UI table
                    self._update_table()
                elif self.camDetails['model_used'] == 'YOLOv11DetectionModel':
                    frameDict1= self.model.predict(frameDict)
                    res_frame = self.model.plot_bounding_boxes(frame,frameDict1,self.cam_name)
                    self._update_table()
                else:
                    res_frame = frame
                _, buffer = cv2.imencode(".jpg", res_frame)
                img_str = base64.b64encode(buffer).decode("utf-8")

                self.streaming_window.content.src_base64 = f"{img_str}"
                self.streaming_window.update()

            except Exception as e:
                print(f"Error processing frame: {e}")
                break

    async def _disconnect(self, e):
        """Disconnect from the camera source"""
        self.connect_button.text = f"Connect {self.cam_name}"
        self.connect_button.style.bgcolor = {
            ft.MaterialState.DEFAULT: ft.colors.GREEN,
            ft.MaterialState.HOVERED: ft.colors.GREEN_700,
        }
        self.connect_button.update()

        if self.cap and self.cap.isOpened():
            self.cap.release()

        if self.type in ['ptz', 'ptz_fixed']:
            if hasattr(self, 'websocket') and self.websocket:
                await self.websocket.close()
            if hasattr(self, 'session') and self.session:
                await self.session.close()

        self.streaming_window.content.src = "assets/disconnected.png"
        self.streaming_window.update()

    async def _toggle_connection(self, e):
        """Toggle camera connection"""
        self.is_connected = not self.is_connected
        if self.is_connected:
            await self._connect(e)
        else:
            await self._disconnect(e)

    def build(self):
        """Build the camera window UI"""
        controls = [
            ft.Text(self.cam_name, size=20, weight=ft.FontWeight.NORMAL),
            self.streaming_window,
            self.connect_button
        ]

        if self.ptz_controls:
            controls.append(self.ptz_controls)
        
        if self.anpr_table:
            controls.append(self.anpr_table)

        if self.obj_table:
            controls.append(self.obj_table)

        return ft.Container(
            content=ft.Column(
                controls=controls,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
                expand=True,
            ),
            padding=20,
            border_radius=10,
            bgcolor=ft.colors.SURFACE,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=ft.colors.with_opacity(0.3, ft.colors.SHADOW)
            ),
            expand=True,
        )