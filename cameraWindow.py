import flet as ft
import cv2
import base64
import threading, sqlite3
from modelFactory import ANPRModel, YOLOv11SegmentationModel, YOLOv11DetectionModel  # Import the ML models
import requests
from websocket import create_connection
import aiohttp
import asyncio

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

        self.websocket = None
        self.ptz_active = {
        "wiper": False,
        "left": False,
        "right": False,
        "up": False,
        "down": False,
        "zoom_in": False,
        "zoom_out": False,
    }

        
    
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
                rows=[
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text("")),
                            ft.DataCell(ft.Text("")),
                            ft.DataCell(ft.Text("")),
                            ft.DataCell(ft.Text("")),
                        ]
                    ) for _ in range(3)
                ],
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
                    rows=[
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text("")),
                                ft.DataCell(ft.Text("")),
                                ft.DataCell(ft.Text("")),
                            ]
                        ) for _ in range(3)
                    ],
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
        """Fetch the latest 3 records from the database"""
        with sqlite3.connect('records.db') as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * 
                FROM {self.cam_name} 
                ORDER BY ROWID DESC 
                LIMIT 3
            """)
            return cursor.fetchall()
        
    def _update_table(self,source_id):
        """Update the table with latest records"""
        records = self._fetch_latest_records()
        while len(records)<5:
            records.append(("--","--","--","--") if self.type=='Anpr' else ("--","--","--"))

        table=self.data_tables[source_id]
        for i, record in enumerate(records[:3]):
            if self.type=='Anpr':
                table.rows[i].cells[0].content.value=str(record[0]) #Time
                table.rows[i].cells[1].content.value=record[1] #ID
                table.rows[i].cells[2].content.value=record[2] #Type
                table.rows[i].cells[3].content.value=record[3] #License NUmber
            else:
                table.rows[i].cells[0].content.value=str(record[0]) #Time
                table.rows[i].cells[1].content.value=record[1] #ID
                table.rows[i].cells[2].content.value=record[2] #Type
        #Refresh table to apply changes
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
    
    # ORIGINAL PTZ BUTTONS CODE
    # def create_ptz_controls(self,source_id):
    #     return ft.Column([
    #         ft.Row([
    #             ft.IconButton(ft.icons.ARROW_UPWARD, icon_color=ft.colors.BLUE),
    #             ft.IconButton(ft.icons.ARROW_DOWNWARD, icon_color=ft.colors.BLUE),
    #             ft.IconButton(ft.icons.ARROW_BACK, icon_color=ft.colors.BLUE),
    #             ft.IconButton(ft.icons.ARROW_FORWARD, icon_color=ft.colors.BLUE),
    #             ft.IconButton(ft.icons.SWIPE, icon_color=ft.colors.RED),
    #             ft.Slider(min=0, max=100, divisions=10, label="{value}%", expand=True),
    #         ], alignment=ft.MainAxisAlignment.CENTER),
    #     ])

    def create_ptz_controls(self, source_id):
    # Helper to create a button with an action
        def create_button(icon, color, action):
            return ft.IconButton(
                icon,
                icon_color=color,
                on_click=lambda e: asyncio.run(self.toggle_ptz_action(action, e))  # Attach PTZ control function
            )

        # Create the PTZ control column
        return ft.Column([
            ft.Row([
                create_button(ft.icons.ARROW_UPWARD, ft.colors.BLUE, "up"),       # Up button
                create_button(ft.icons.ARROW_DOWNWARD, ft.colors.BLUE, "down"),   # Down button
                create_button(ft.icons.ARROW_BACK, ft.colors.BLUE, "left"),       # Left button
                create_button(ft.icons.ARROW_FORWARD, ft.colors.BLUE, "right"),   # Right button
                create_button(ft.icons.SWIPE, ft.colors.RED, "wiper"),            # Wiper button
                ft.Slider(
                    min=0,
                    max=100,
                    divisions=10,
                    label="{value}%",
                    expand=True,
                    on_change=lambda e: asyncio.run(self.handle_zoom_change(e.value))  # Attach zoom handling
                ),
            ], alignment=ft.MainAxisAlignment.CENTER),
        ])



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
    
    # TOGGLE CONNECTION
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


    
    # CONNECTION FUNCTION
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

        # Establish WebSocket connection if it's a PTZ camera
        if self.cam_details.get('type') in ['ptz', 'ptz_fixed']:
            base_url = self.cam_details['base_url']
            url = self.cam_details['url']
            asyncio.run(self.establish_ptz_connection(base_url, url))

        # Store the video capture object in the connections dictionary
        self.connections[source_id]["cap"] = cap

        # Start a background thread to read frames
        threading.Thread(target=self.read_frames, args=(source_id,), daemon=True).start()

    # ESTABLISH PTZ CONNECTION FUNCTION
    async def establish_ptz_connection(self, base_url, url):
        try:
            session_cookie = await self.get_session_cookie(base_url)
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
            print(f"Successfully connected to PTZ WebSocket: {url}")
            self.is_connected = True
        except Exception as e:
            print(f"Failed to establish PTZ WebSocket connection: {e}")
            self.is_connected = False

    # GET SESSION COOKIE 
    async def get_session_cookie(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                cookies = session.cookie_jar.filter_cookies(url)
                session_cookie = cookies.get('PHPSESSID')
                return session_cookie.value if session_cookie else ""

    # DISCONNECT FUNCTION
    def disconnect(self, connect_button, source_id):
        # Update button to "Connect"
        connect_button.text = "Connect"
        connect_button.style.bgcolor = {
            ft.MaterialState.DEFAULT: ft.colors.GREEN,
            ft.MaterialState.HOVERED: ft.colors.GREEN_700,
        }
        connect_button.update()

        # Close WebSocket and session for PTZ camera
        if self.cam_details.get('type') in ['ptz', 'ptz_fixed']:
            asyncio.run(self.close_ptz_connection())

        connection = self.connections.get(source_id, None)
        if connection and connection["cap"]:
            connection["cap"].release()
            connection["cap"] = None

        # Update streaming window to show disconnected state
        window = self.streaming_windows.get(source_id, None)
        if window:
            window.content.src = "assets/disconnect.png"
            window.update()

    # CLOSE PTZ CONNECTION
    async def close_ptz_connection(self):
        if self.websocket:
            await self.websocket.close()
        if self.session:
            await self.session.close()
    
    # READ FRAMES FUNCTION
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
    
    # PTZ_CAM_CONTROL FUNCTIONS
    async def send_message(self, message):
        try:
            if self.websocket and not self.websocket.closed:
                await self.websocket.send_str(message)
                print(f"Sent message: {message}")
            else:
                print("WebSocket is not connected or has been closed")
        except Exception as e:
            print(f"Failed to send message: {e}")

    async def toggle_ptz_action(self, action, e):
        self.ptz_active[action] = not self.ptz_active[action]
        message = self.get_ptz_message(action, self.ptz_active[action])
        await self.send_message(message)

    def get_ptz_message(self, action, active):
        messages = {
            "wiper": "type=ptz&aux_on=2" if active else "type=ptz&aux_on=0",
            "left": "type=ptz&move=left&pspd=30" if active else "type=ptz&move=stop",
            "right": "type=ptz&move=right&pspd=30" if active else "type=ptz&move=stop",
            "up": "type=ptz&move=up&pspd=30" if active else "type=ptz&move=stop",
            "down": "type=ptz&move=down&pspd=30" if active else "type=ptz&move=stop",
            "zoom_in": "type=ptz&zoom=tele&zspd=5" if active else "type=ptz&zoom=stop"
        }
        return messages.get(action, "Invalid action!!!")
    
    # async def toggle_left(self,action,e):
    #     self.ptz_active[action] = not self.ptz_active[action]
    #     message = self.get_ptz_message(action, self.ptz_active[action])
    #     await self.send_message(message)

    # async def toggle_right(self,action,e):
    #     self.ptz_active[action] = not self.ptz_active[action]
    #     message = self.get_ptz_message(action, self.ptz_active[action])
    #     await self.send_message(message)

    # async def toggle_up(self,action,e):
    #     self.ptz_active[action] = not self.ptz_active[action]
    #     message = self.get_ptz_message(action, self.ptz_active[action])
    #     await self.send_message(message)

    # async def toggle_down(self,action,e):
    #     self.ptz_active[action] = not self.ptz_active[action]
    #     message = self.get_ptz_message(action, self.ptz_active[action])
    #     await self.send_message(message)

    # async def toggle_wiper(self,action,e):
    #     self.ptz_active[action] = not self.ptz_active[action]
    #     message = self.get_ptz_message(action, self.ptz_active[action])
    #     await self.send_message(message)


    # *********************************************************************


    def build(self, source_id, cam_name):
        streaming_window = self.create_streaming_window()
        table = self.create_table(source_id)
        self.streaming_windows[source_id] = streaming_window
        connect_button = self.create_connect_button(source_id)
        ptz_controls=None
        if self.cam_details.get('type') in ['ptz', 'ptz_fixed']:
            ptz_controls=self.create_ptz_controls(source_id)

        controls = [
                    ft.Text(
                        value=cam_name,  # Display the camera name
                        style=ft.TextStyle(size=16, weight="bold"),  # Bold, larger font
                    ),
                    streaming_window,  # Streaming window
                    connect_button,  # Connect button
                    ptz_controls,
                    table
        ]

         # Remove any None values from the list
        controls = [control for control in controls if control is not None]

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

    
    