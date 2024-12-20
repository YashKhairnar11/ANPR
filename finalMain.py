import flet as ft
import yaml
import logging
from cameraWindow import WindowStreamer
from cameraSelector import CameraSelector

############ The entire application #############################
class Application():
    def __init__(self, config, page):
        self.page = page
        self.config = config
        self.camera_windows = {}
        self.selected_cameras = []

    def load_cameras(self):
        cameras = self.config.get('cameras', [])
        for cam_config in cameras:
            for cam_name, cam_details in cam_config.items():
                self.camera_windows[cam_name] = WindowStreamer(cam_name, cam_details)
        print(self.camera_windows)      

    def on_camera_selection_change(self, selected_cameras):
        self.selected_cameras = selected_cameras
        print(self.selected_cameras)
        self.update_grid_layout()


    def create_camera_selector(self):
        return CameraSelector(list(self.camera_windows.keys()), self.on_camera_selection_change)


    def create_grid_layout(self):
        if not self.selected_cameras:
            return self.create_no_cameras_card()
        
        grid_rows = []
        for i in range(0, len(self.selected_cameras), 2):
            row_controls = [self.camera_windows[cam].build(self.camera_windows[cam].cam_details['source'],cam) for cam in self.selected_cameras[i:i+2]]
            row = ft.Row(
                controls=row_controls,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=20,
                expand=True,
            )
            grid_rows.append(row)

        print(len(grid_rows))  
        return ft.Column(
            controls=grid_rows,
            spacing=20,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
        )

    def create_no_cameras_card(self):
        return ft.Container(
            content=ft.Column(
                [
                    ft.Icon(ft.icons.VIDEOCAM_OFF, size=64, color=ft.colors.GREY_400),
                    ft.Text("No Cameras Selected", size=24, weight=ft.FontWeight.BOLD),
                    ft.Text("Please select up to 4 cameras from the list on the left.", size=16),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=20,
            ),
            alignment=ft.alignment.center,
            padding=50,
            border_radius=10,
            bgcolor=ft.colors.SURFACE,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=ft.colors.with_opacity(0.3, ft.colors.SHADOW)
            ),
            expand=True,
        )

    def update_grid_layout(self):
        if not hasattr(self, 'grid_container'):
            print("Grid container not initialized.")
            return

        grid_layout = self.create_grid_layout()
        if grid_layout:
            self.grid_container.content = grid_layout
            self.grid_container.update()
        self.page.update()
            
        
    def build(self):
        self.camera_selector = self.create_camera_selector()
        self.grid_container = ft.Container(
            content=self.create_grid_layout(),
            expand=True,
        )
        return ft.Column([
            ft.Row([
                self.camera_selector.build(),
                self.grid_container
            ], alignment=ft.MainAxisAlignment.START, spacing=20, expand=True)
        ], spacing=20, expand=True)



def main(page: ft.Page):
    try:
        try:
            with open('config.yaml', 'r') as file:
                config = yaml.safe_load(file)
        except (FileNotFoundError, IOError):
            print("No config.yaml found. Using default configuration.")

        page.title = "Accurate Vision Intelli System"
        page.padding = 20
        page.theme_mode = ft.ThemeMode.DARK
        page.theme = ft.Theme(color_scheme_seed=ft.colors.BLUE)
        

        app = Application(config,page)
        app.load_cameras()

        page.add(app.build())
        page.update()

    except Exception as e:
        logging.error(f"Application initialization error: {e}")

if __name__ == "__main__":
    ft.app(target=main)

    