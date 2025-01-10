import flet as ft
############ Camera Selector ################################
class CameraSelector:
    def __init__(self, cameras, on_camera_selection_change):
        self.cameras = cameras
        self.on_camera_selection_change = on_camera_selection_change
        self.checkboxes = []

    def build(self):
        for cam_name in self.cameras:
            checkbox = ft.Checkbox(
                label=cam_name,
                label_style=ft.TextStyle(color=ft.colors.BLACK, weight=ft.FontWeight.BOLD),  # Change label color to red
                value=False,
                on_change=self.handle_checkbox_change,
                check_color=ft.colors.GREEN,
                fill_color=ft.colors.BLACK,
            )
            self.checkboxes.append(checkbox)

        img =ft.Image(
            src='assets/group-logo.png',
            width=250,
            height=175,
            fit=ft.ImageFit.CONTAIN
        )

        return ft.Container(
            content=ft.Column(
                controls=[
                    img,
                    ft.Text("Select Cameras (Max 4):", size=16, weight=ft.FontWeight.BOLD, color=ft.colors.BLACK),
                    *self.checkboxes
                ],
                spacing=10
            ),
            padding=20,
            border_radius=10,
            bgcolor=ft.colors.WHITE,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=ft.colors.with_opacity(0.3, ft.colors.SHADOW)
            ),
            width=250,  # Fixed width for the selector
        )

    def handle_checkbox_change(self, e):
        selected_count = sum(1 for cb in self.checkboxes if cb.value)
        if selected_count > 4:
            e.control.value = False
            e.control.update()
        else:
            selected_cameras = [cb.label for cb in self.checkboxes if cb.value]
            self.on_camera_selection_change(selected_cameras)
