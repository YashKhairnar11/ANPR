import flet as ft
import aiohttp
import threading
from modelFactory import ANPRModel,YOLOv11SegmentationModel,YOLOv11DetectionModel
import base64, cv2

############### Camera window ###########################
class CameraWindow:
    def __init__(self, cam_name, cam_details):
        self.camDetails = cam_details
        self.cam_name = cam_name
        self.type = cam_details['type']
        self.source = cam_details['source']
        self.cap = None
        self.thread = None
        self.model = self.initialize_model()
        self.is_connected = False

        #streaming window
        self.streaming_window = ft.Container(
            width=720,
            height=480,
            content=ft.Image(
                src="assets/disconnect.png",
                width=50,
                height=50,
                fit=ft.ImageFit.CONTAIN,  # Added image fitting
            ),
            border=ft.border.all(2, ft.colors.OUTLINE),
            margin=ft.margin.all(10),  # Added margin
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            expand=True,  # Allow the container to expand
        )

        #connect button
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
            on_click=self.toggle_connection
        )
        
        self.ptz_controls = self.create_ptz_controls() if self.type in ['ptz', 'ptz_fixed'] else None


    # function to initialize the source for a window
    def initialize_model(self):
        if self.camDetails['model_used']=='ANPRModel':
            return ANPRModel()
        elif self.camDetails['model_used']=='YOLOv11DetectionModel':
            return YOLOv11DetectionModel()
        elif self.camDetails['model_used']=='YOLOv11SegmentationModel':
            return YOLOv11SegmentationModel()
        else:
            return None
        
    def create_ptz_controls(self):
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


    ## async function to connect to rtsp websocket ############
    async def get_session_cookie(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                cookies = session.cookie_jar.filter_cookies(url)
                session_cookie = cookies.get('PHPSESSID')
                return session_cookie.value if session_cookie else ""
    
    # connect to source and start the feed #############
    async def connect(self,e):
        e = self.connect_button
        e.text = f"Disconnect {self.cam_name}"
        e.style.bgcolor = {
                ft.MaterialState.DEFAULT: ft.colors.RED,
                ft.MaterialState.HOVERED: ft.colors.RED_700, 
            }
        e.update()
        
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"Failed to open video source: {self.source}")
        # when the cameras are ptz
        if self.type in ['ptz', 'ptz_fixed']:
            base_url = self.camDetails['base_url']
            url = self.camDetails['url']
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

            except aiohttp.ClientConnectorError as e:
                print(f"Connection error: {e}")
                self.is_connected = False
            except Exception as e:
                print(f"Failed to connect: {type(e).__name__}: {e}")
                self.is_connected = False
        # start the feed capturing and rendering
        self.start_video_stream()


    def start_video_stream(self):
        self.thread = threading.Thread(target=self.read_frames, daemon=True).start()


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
            
            res_frame = 0
            try:
                if self.camDetails['model_used'] == 'ANPRModel':
                    frameDict1 = self.model.det_objects(frameDict)
                    frameDict2 = self.model.det_plates_ocr(frameDict1)
                    res_frame = self.model.plot_bounding_boxes(frame,frameDict2)
                elif self.camDetails['model_used'] == 'YOLOv11DetectionModel':
                    res_frame = self.model.predict(frame)
                else:
                    res_frame = frame

                _, buffer = cv2.imencode(".jpg", res_frame)
                img_str = base64.b64encode(buffer).decode("utf-8")

                self.streaming_window.content.src_base64 = f"{img_str}"
                self.streaming_window.update()

            except Exception as e:
                print(f"Error processing frame: {e}")
                break


    async def disconnect(self,e):
        e = self.connect_button
        e.text = f"Connect {self.cam_name}"
        e.style.bgcolor = {
                ft.MaterialState.DEFAULT: ft.colors.GREEN,
                ft.MaterialState.HOVERED: ft.colors.GREEN_700,
            }
        e.update()

        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.type in ['ptz', 'ptz_fixed']:
            if hasattr(self, 'websocket') and self.websocket:
                await self.websocket.close()
            if hasattr(self, 'session') and self.session:
                await self.session.close()
        self.streaming_window.content.src = "assets/disconnected.png"
        self.streaming_window.update()
        

    async def toggle_connection(self, e):
        self.is_connected = not self.is_connected
        if self.is_connected :
            await self.connect(e)
        else:
            await self.disconnect(e)
            
    def build(self):
        print(self.connect_button)
        controls = [
            self.connect_button,
            ft.Text(self.cam_name, size=20, weight=ft.FontWeight.NORMAL),
            self.streaming_window
          
        ]
        if self.ptz_controls:
            controls.append(self.ptz_controls)
        
        return ft.Container(
            content=ft.Column(
                controls=controls,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
                expand=True,  # Allow the column to expand
            ),
            padding=20,
            border_radius=10,
            bgcolor=ft.colors.SURFACE,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=ft.colors.with_opacity(0.3, ft.colors.SHADOW)
            ),
            expand=True,  # Allow the container to expand
        )

