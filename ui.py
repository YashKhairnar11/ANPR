import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QSlider, QLabel, QFrame
)
from PyQt5.QtCore import Qt
import yaml
import math

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Camera Control Interface")
        self.setGeometry(100, 100, 1200, 600)

        # Main layout to divide left and right sections
        main_layout = QHBoxLayout()

        # Left control panel
        control_panel = self.create_control_panel()
        main_layout.addLayout(control_panel, 1)  # Left panel takes up less space

        # Right video stream area
        video_panel = self.create_video_panel()
        main_layout.addLayout(video_panel, 3)  # Right panel takes up more space

        # Set main layout to central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_control_panel(self):
        """Create the left control panel with directional buttons and zoom slider."""
        layout = QVBoxLayout()

        # Label for control panel
        layout.addWidget(QLabel("Controls"))

        # Create a grid layout for the directional controls
        grid_layout = QGridLayout()

        # Directional buttons
        up_button = QPushButton("Up")
        down_button = QPushButton("Down")
        left_button = QPushButton("Left")
        right_button = QPushButton("Right")
        wiper_button = QPushButton("Wiper")

        # Arrange buttons in a 3x3 grid
        grid_layout.addWidget(up_button, 0, 1)
        grid_layout.addWidget(left_button, 1, 0)
        grid_layout.addWidget(wiper_button, 1, 1)  # Center button
        grid_layout.addWidget(right_button, 1, 2)
        grid_layout.addWidget(down_button, 2, 1)

        # Add the grid layout to the main control layout
        layout.addLayout(grid_layout)

        # Zoom slider
        zoom_label = QLabel("Zoom")
        layout.addWidget(zoom_label)

        zoom_slider = QSlider(Qt.Horizontal)
        zoom_slider.setMinimum(0)
        zoom_slider.setMaximum(100)
        layout.addWidget(zoom_slider)

        # Spacer for alignment
        layout.addStretch()
        
        return layout

    def create_video_panel(self):
        """Create the right video panel for displaying camera feeds."""
        layout = QGridLayout()  # Grid layout for multiple camera views
        
        # Try loading camera configuration from YAML
        try:
            with open('camConfig.yaml', 'r') as file:
                config = yaml.safe_load(file)
            cameras = config.get('cameras', {})
            
            num_cameras = len(cameras)
            columns = 2  
            rows = math.ceil(num_cameras / columns)

            # Create a QLabel for each camera in the config
            for index, (camera_id, camera_info) in enumerate(cameras.items()):
                row = index // columns
                col = index % columns
                camera_view = QLabel(camera_info.get('name', f"Camera {index + 1}"))
                camera_view.setFrameStyle(QFrame.Box | QFrame.Plain)
                camera_view.setStyleSheet("background-color: black; color: white;")
                camera_view.setAlignment(Qt.AlignCenter)
                camera_view.setMinimumSize(300, 200)
                layout.addWidget(camera_view, row, col)

        except Exception as e:
            # Handle errors with a fallback
            error_label = QLabel("Error loading cameras from config.")
            error_label.setStyleSheet("color: red;")
            layout.addWidget(error_label)

        return layout

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
