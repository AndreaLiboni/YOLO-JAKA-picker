import sys
sys.path.append('/home/andrea/Sviluppo/Lavoro/jaka_sdk_2.2.7')
from jkrc import RC

import cv2
import numpy as np
import json
import platform
import time
import os
import yaml
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QFont

from yolo import YOLOModel


class ClickableLabel(QLabel):
    """Custom QLabel that emits signals when clicked"""
    clicked = pyqtSignal(int, int)
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid gray;")
        self.setAlignment(Qt.AlignCenter)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = event.y()
            self.clicked.emit(x, y)


class CameraApp(QMainWindow):
    CONFIGS = yaml.safe_load(open('config.yaml'))

    def __init__(self):
        super().__init__()
        self.camera = None
        self.timer = QTimer()
        self.current_frame = None
        self.processed_frame = None
        self.frame_size = (640, 480)  # Default frame size
        self.new_frame_size = None
        self.origin = (0, 0)
        self.points = []
        
        # Charuco board parameters
        self.squares_x = 11 # horizontal squares
        self.squares_y = 8  # vertical squares
        self.square_length = 15
        self.marker_length = 11
        self.work_area_size = self.CONFIGS['camera']['work_area_size']  # in mm

        
        # Create Charuco board
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.charuco_board = cv2.aruco.CharucoBoard(
            size=(self.squares_x, self.squares_y), 
            squareLength=self.square_length,
            markerLength=self.marker_length, 
            dictionary=self.aruco_dict
        )
        self.charuco_params = cv2.aruco.DetectorParameters()
        self.charuco_board.setLegacyPattern(True)
        
        # Calibration data
        self.calibration_corners = []
        self.calibration_ids = []
        self.camera_matrix = None  # Camera intrinsic parameters
        self.distortion = None
        self.rvec = None
        self.tvec = None
        self.is_calibrated = False
        self.show_undistorted = False
        self.show_cropped = False
        self.img_folder = "./calibration_images"
        self.margin_rectified = 10
        self.size_rectified = None
        self.H = None

        self.yolo = None
        self.points_detection = []
        self.classes = self.CONFIGS['model']['classes']

        self.is_picking_enabled = False
        
        self.init_ui()
        self.init_camera()
        self.start_camera()

        try:
            self.load_calibration()
        except Exception:
            pass
        
    def init_ui(self):
        self.setWindowTitle("Dual Camera Stream with Charuco Calibration")
        self.setGeometry(50, 100, 1600, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Right side - Interactive frame and controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Interactive stream frame with Charuco detection
        interactive_frame = QGroupBox()
        interactive_layout = QVBoxLayout(interactive_frame)
        
        self.interactive_label = ClickableLabel()
        self.interactive_label.setText("Please start the camera")
        self.interactive_label.clicked.connect(self.on_point_clicked)
        interactive_layout.addWidget(self.interactive_label)
        
        # Status label for Charuco detection
        self.charuco_status = QLabel("Charuco Status: Not detected")
        self.charuco_status.setStyleSheet("color: red; font-weight: bold;")
        interactive_layout.addWidget(self.charuco_status)
        
        # Controls frame
        controls_frame = QGroupBox()
        controls_layout = QVBoxLayout(controls_frame)
        
        # Camera control buttons
        camera_button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")
        
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        
        camera_button_layout.addWidget(self.start_btn)
        camera_button_layout.addWidget(self.stop_btn)
        
        # Calibration control buttons
        calib_button_layout = QHBoxLayout()
        self.capture_btn = QPushButton("Capture Calibration Frame")
        self.calibrate_btn = QPushButton("Calibrate Camera")
        self.load_calib_btn = QPushButton("Load Calibration")
        self.reset_calib_btn = QPushButton("Reset Calibration")
        
        self.capture_btn.clicked.connect(self.capture_calibration_frame)
        self.calibrate_btn.clicked.connect(self.calibrate_camera)
        self.reset_calib_btn.clicked.connect(self.reset_calibration)
        self.load_calib_btn.clicked.connect(self.load_calibration)
        
        calib_button_layout.addWidget(self.capture_btn)
        calib_button_layout.addWidget(self.calibrate_btn)
        calib_button_layout.addWidget(self.reset_calib_btn)
        calib_button_layout.addWidget(self.load_calib_btn)
        
        # Undistortion toggle
        self.undistort_checkbox = QCheckBox("Show Undistorted Image")
        self.undistort_checkbox.toggled.connect(self.toggle_undistortion)

        # Crop toggle
        self.crop_checkbox = QCheckBox("Crop Image")
        self.crop_checkbox.toggled.connect(self.toggle_crop)

        # Enable detection
        self.enable_detection_checkbox = QCheckBox("Enable Detection")
        self.enable_detection_checkbox.toggled.connect(self.toggle_detection)

        # Start picking
        self.enable_picking_checkbox = QCheckBox("Enable Picking (will use the robot)")
        self.enable_picking_checkbox.toggled.connect(self.toggle_picking)
        
        # Point selection controls
        point_button_layout = QHBoxLayout()
        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self.clear_points)
        point_button_layout.addWidget(self.clear_points_btn)
        
        # Points list
        self.points_label = QLabel("Selected Points:")
        self.points_list = QListWidget()
        self.points_list.setMaximumHeight(300)
        
        # Add all controls to layout
        controls_layout.addLayout(camera_button_layout)
        controls_layout.addLayout(calib_button_layout)
        controls_layout.addWidget(self.undistort_checkbox)
        controls_layout.addWidget(self.crop_checkbox)
        controls_layout.addWidget(self.enable_detection_checkbox)
        controls_layout.addWidget(self.enable_picking_checkbox)
        controls_layout.addWidget(QLabel(f"Calibration Frames Captured: 0"))
        self.frames_label = controls_layout.itemAt(controls_layout.count()-1).widget()
        controls_layout.addWidget(self.points_label)
        controls_layout.addWidget(self.points_list)
        controls_layout.addLayout(point_button_layout)
        
        # Add frames to right layout
        right_layout.addWidget(interactive_frame)
        
        # Add main frames to main layout
        main_layout.addWidget(right_widget, 1)
        main_layout.addWidget(controls_frame, 1)
        
        # Set initial button states
        self.stop_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.calibrate_btn.setEnabled(self.img_folder is not None)
        self.undistort_checkbox.setEnabled(False)
        
    def init_camera(self):
        """Initialize camera connection"""
        try:
            index = self.CONFIGS['camera']['index']
            print("Platform: ", platform.system())
            if platform.system() == "Windows":
                #sets the Windows cv2 backend to DSHOW (Direct Video Input Show)
                cv2.CAP_DSHOW
                self.camera = cv2.VideoCapture(index)
            elif platform.system() == "Linux":
                # set the Linux cv2 backend to GTREAMER
                cv2.CAP_GSTREAMER
                self.camera = cv2.VideoCapture(index)
            else:
                self.camera = cv2.VideoCapture(index)
                # cap.set(cv2.CAP_FFMPEG, cv2.CAP_FFMPEG_VIDEOTOOLBOX) # not sure!
            
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
            
            # Set camera properties to maximum resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


            self.frame_size = (
                int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            print(f"Camera initialized with resolution: {self.frame_size[0]}x{self.frame_size[1]}")

            # Set the live label size to match camera resolution aspect ratio, but with a maximum width
            self.interactive_label.setFixedSize(
                1280, 
                int((1280 * self.frame_size[1]) / self.frame_size[0]) if self.frame_size[0] > 0 else 480
            )
            
        except Exception as e:
            QMessageBox.warning(self, "Camera Error", f"Failed to initialize camera: {str(e)}")
            
    def start_camera(self):
        """Start camera streaming"""
        if self.camera and self.camera.isOpened():
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # Update every 30ms (~33 FPS)
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.capture_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "Camera Error", "Camera not available")
            
    def stop_camera(self):
        """Stop camera streaming"""
        self.timer.stop()
        self.timer.timeout.disconnect()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
    
    def draw_bbox(self, frame):
        cols = self.squares_x
        rows = self.squares_y
        board_width = self.square_length * cols
        board_height = self.square_length * rows

        obj_pts = np.array([
            [0, 0, 0],  # top-left
            [board_width, 0, 0],  # top-right
            [board_width, board_height, 0],  # bottom-right
            [0, board_height, 0]  # bottom-left
        ], dtype=np.float32)

        img_pts, _ = cv2.projectPoints(obj_pts, self.rvec, self.tvec, self.camera_matrix, self.distortion)
        img_pts = img_pts.reshape(-1, 2).astype(np.float32)

        x_min, y_min = np.min(img_pts, axis=0)
        x_max, y_max = np.max(img_pts, axis=0)
        original_bbox = (x_min, y_min, x_max, y_max)
        h, w = frame.shape[:2]

        x_min, y_min, x_max, y_max = map(int, original_bbox)
        x_min -= self.margin_rectified
        y_min -= self.margin_rectified
        x_max += self.margin_rectified
        y_max += self.margin_rectified
        self.size_rectified = (int(x_max - x_min), int(y_max - y_min))
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

        return frame
    

    def rectify_perspective(
        self,
        frame,
        output_width=500
    ):
        cols = self.squares_x
        rows = self.squares_y
        board_width = self.square_length * cols + self.work_area_size[0]
        board_height = self.square_length * rows + self.work_area_size[1]

        # Coordinate dei 4 angoli della board nel mondo (Z=0)
        obj_pts = np.array([
            [0, 0, 0],  # top-left
            [board_width, 0, 0],  # top-right
            [board_width, board_height, 0],  # bottom-right
            [0, board_height, 0]  # bottom-left
        ], dtype=np.float32)


        # Proietta i punti nel piano immagine
        img_pts, _ = cv2.projectPoints(obj_pts, self.rvec, self.tvec, self.camera_matrix, self.distortion)
        img_pts = img_pts.reshape(-1, 2).astype(np.float32)

        # get crop size
        # x_min, y_min = np.min(img_pts, axis=0)
        # x_max, y_max = np.max(img_pts, axis=0)

        # x_min -= self.margin_rectified
        # y_min -= self.margin_rectified
        # x_max += self.margin_rectified
        # y_max += self.margin_rectified

        # self.size_rectified = (int(x_max - x_min), int(y_max - y_min))

        # Definisce le coordinate rettificate in pixel (con margine)
        canvas_inner_width = output_width - 2 * self.margin_rectified
        scale = canvas_inner_width / board_width
        canvas_inner_height = int(board_height * scale)
        output_height = canvas_inner_height + 2 * self.margin_rectified


        dst_pts = np.array([
            [self.margin_rectified, self.margin_rectified],  # top-left
            [self.margin_rectified + canvas_inner_width, self.margin_rectified],  # top-right
            [self.margin_rectified + canvas_inner_width, self.margin_rectified + canvas_inner_height],  # bottom-right
            [self.margin_rectified, self.margin_rectified + canvas_inner_height]  # bottom-left
        ], dtype=np.float32)


        # Calcola omografia da immagine → piano rettificato
        H = cv2.getPerspectiveTransform(img_pts, dst_pts)
        H_inv = np.linalg.inv(H)

        # Applica trasformazione
        rectified_image = cv2.warpPerspective(frame, H, (output_width, output_height))

        return rectified_image, H
    
    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

    
    def update_frame(self):
        # start = time.time()
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.current_frame = frame.copy()

                if self.is_calibrated and self.show_undistorted:
                    # Apply undistortion if calibrated and enabled
                    frame = cv2.undistort(frame, self.camera_matrix, self.distortion)
                
                marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.charuco_params)

                if marker_ids is not None and len(marker_ids) > 0:
                    # Interpolate CharUco corners
                    charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, frame, self.charuco_board)

                    # If enough corners are found, estimate the pose
                    if charuco_retval and len(charuco_corners) > 6:
                        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.charuco_board, self.camera_matrix, self.distortion, None, None)

                        # If pose estimation is successful, draw the axis
                        if retval:
                            self.rvec = rvec
                            self.tvec = tvec

                            cv2.drawFrameAxes(frame, self.camera_matrix, self.distortion, rvec, tvec, length=50, thickness=2)

                            # origin_index = marker_ids.tolist().index(min(marker_ids))
                            # self.origin = (int(marker_corners[origin_index][0][0][0]), int(marker_corners[origin_index][0][0][1]))
                            self.origin, _ = cv2.projectPoints(np.array([[0,0,0]], dtype=np.float32), self.rvec, self.tvec, self.camera_matrix, self.distortion)
                            self.origin = (int(self.origin[0][0][0]), int(self.origin[0][0][1]))

                            if self.is_calibrated and self.show_cropped:
                                frame, H = self.rectify_perspective(frame, output_width=self.interactive_label.width())
                                self.H = H
                                frame = self.image_resize(frame, height=self.interactive_label.height())
                                if self.new_frame_size != frame.shape[:2]:
                                    self.new_frame_size = frame.shape[:2]
                                    self.interactive_label.setFixedSize(
                                        frame.shape[1],
                                        self.interactive_label.height()
                                    )
                                

                                self.origin = cv2.perspectiveTransform(np.array([[[self.origin[0], self.origin[1]]]], dtype=np.float32), H)[0][0]
                                self.origin = (int(self.origin[0]), int(self.origin[1]))
                                
                                # marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(frame_no_origin, self.aruco_dict, parameters=self.charuco_params)
                                # origin_index = marker_ids.tolist().index(min(marker_ids))
                                # self.origin = (int(marker_corners[origin_index][0][0][0]), int(marker_corners[origin_index][0][0][1]))
                            
                            else:
                                frame = self.draw_bbox(frame)
                # detection
                boxes_detection = None
                if self.yolo is not None:
                    boxes_detection = self.yolo.forward(frame.copy())
                

                if boxes_detection is not None:
                    self.points_detection = []
                    for box in boxes_detection:
                        x1, y1, x2, y2, cl = map(round, box)

                        real_x, real_y = map(round, self.img_to_realworld((x1 + x2)//2, (y1 + y2)//2))
                        dist = round(np.sqrt(real_x**2 + real_y**2))
                        self.points_detection.append((real_x, real_y))

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{self.classes[cl]} dist: {dist}mm ({real_x}, {real_y})', (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)




                display_frame = frame.copy()
                
                # Update interactive display
                self.processed_frame = display_frame.copy()
                self.update_interactive_display(display_frame)
        # end = time.time()
        # print("interval: ", end - start, " seconds")
        
    def update_interactive_display(self, frame):
        """Update the interactive display with points"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        
        # Draw points
        pixmap = self.draw_points_on_image(pixmap)
            
        scaled_pixmap = pixmap.scaled(self.interactive_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.interactive_label.setPixmap(scaled_pixmap)
    
    def pixel_to_meter(self, pixel_value):
        return pixel_value * self.tvec[2][0] / self.camera_matrix[0][0]

    def meter_to_pixel(self, meter_value):
        return meter_value * self.camera_matrix[0][0] / self.tvec[2][0]
        
    def capture_calibration_frame(self):
        if self.img_folder is not None:
            # Save the current frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.img_folder}/calibration_frame_{timestamp}.png"
            cv2.imwrite(filename, self.current_frame)
            
        if self.processed_frame is not None:
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
            
            if ids is not None and len(ids) > 0:
                charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.charuco_board)
                
                if charuco_ids is not None and len(charuco_corners) > 3:  # Need at least 4 corners for calibration
                    self.calibration_corners.append(charuco_corners)
                    self.calibration_ids.append(charuco_ids)
                    
                    if len(self.calibration_corners) >= 3:
                        self.calibrate_btn.setEnabled(True)
                    
                else:
                    QMessageBox.warning(self, "Insufficient Corners", 
                                      "Need at least 4 Charuco corners visible for calibration")
            else:
                QMessageBox.warning(self, "No Charuco Detected", 
                                  "Please ensure Charuco board is visible and well-lit")

    def calibrate_and_save_parameters(self):
        # Define the aruco dictionary and charuco board
        params = cv2.aruco.DetectorParameters()

        # Load PNG images from folder
        image_files = [os.path.join(self.img_folder, f) for f in os.listdir(self.img_folder) if f.endswith(".png")]
        image_files.sort()  # Ensure files are in order

        all_charuco_corners = []
        all_charuco_ids = []

        for image_file in image_files:
            image = cv2.imread(image_file)
            image_copy = image.copy()
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=params)
            
            # If at least one marker is detected
            if len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
                charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, self.charuco_board)
                if charuco_retval:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)

        # Calibrate camera
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, self.charuco_board, image.shape[:2], None, None)

        # Save calibration data
        # np.save('camera_matrix.npy', camera_matrix)
        # np.save('dist_coeffs.npy', dist_coeffs)
        self.camera_matrix = camera_matrix
        self.distortion = dist_coeffs
        self.is_calibrated = True
        self.undistort_checkbox.setEnabled(True)
        self.save_calibration()
    
        
    def save_calibration(self):
        OUTPUT_JSON = 'calibration.json'

        data = {
            "mtx": self.camera_matrix.tolist(),
            "dist": self.distortion.tolist(),
        }

        with open(OUTPUT_JSON, 'w') as json_file:
            json.dump(data, json_file, indent=4)


    def calibrate_camera(self):
        """Perform camera calibration using captured frames"""
        if self.img_folder is not None:
            return self.calibrate_and_save_parameters()
        if len(self.calibration_corners) < 3:
            QMessageBox.warning(self, "Insufficient Data", 
                              "Need at least 3 calibration frames")
            return
            
        try:
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
            
            # Get image size
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            img_size = gray.shape[::-1]
            
            # Calibrate camera
            ret, self.camera_matrix, self.distortion, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                self.calibration_corners, self.calibration_ids, self.charuco_board, img_size, None, None
            )
            
            if ret:
                self.is_calibrated = True
                self.undistort_checkbox.setEnabled(True)
                self.rvec = rvecs[0]
                self.tvec = tvecs[0]
                self.save_calibration()
                
                QMessageBox.information(self, "Calibration Complete", 
                                      f"Camera calibration successful!\nRMS Error: {ret:.4f}")
            else:
                QMessageBox.warning(self, "Calibration Failed", 
                                  "Camera calibration failed. Try capturing more frames.")
                
        except Exception as e:
            QMessageBox.critical(self, "Calibration Error", f"Error during calibration: {str(e)}")
            
    def reset_calibration(self):
        """Reset calibration data"""
        self.calibration_corners.clear()
        self.calibration_ids.clear()
        self.camera_matrix = None
        self.distortion = None
        self.is_calibrated = False
        self.show_undistorted = False
        self.show_cropped = False
        self.rvec = None
        self.tvec = None
        
        self.frames_label.setText("Calibration Frames Captured: 0")
        self.calibrate_btn.setEnabled(False)
        self.undistort_checkbox.setEnabled(False)
        self.undistort_checkbox.setChecked(False)
        
        QMessageBox.information(self, "Reset Complete", "Calibration data has been reset")
    
    def load_calibration(self):
        # get frame
        if self.camera is None or not self.camera.isOpened():
            QMessageBox.warning(self, "Camera Error", "Camera not available")
            return
        ret, frame = self.camera.read()
        if not ret:
            QMessageBox.warning(self, "Camera Error", "Failed to read from camera")
            return
        img_size = frame.shape[1::-1]  # (width, height)

        json_file_path = './calibration.json'

        with open(json_file_path, 'r') as file: # Read the JSON file
            json_data = json.load(file)

        self.camera_matrix = np.array(json_data['mtx'])
        self.distortion = np.array(json_data['dist'])
    
        self.is_calibrated = True
        self.undistort_checkbox.setEnabled(True)

        
    def toggle_undistortion(self, checked):
        self.show_undistorted = checked
    
    def toggle_crop(self, checked):
        self.show_cropped = checked
        if not self.show_cropped:
            self.interactive_label.setFixedSize(
                self.frame_size[0],
                self.interactive_label.height()
            )
    
    def toggle_detection(self, checked):
        if checked and self.yolo is None:
            self.yolo = YOLOModel(
                model_path=self.CONFIGS['model']['model_path'],
                device=self.CONFIGS['model']['device'],
                conf_thres=self.CONFIGS['model']['confidence_threshold'],
            )
        elif not checked:
            self.yolo = None
            self.points_detection.clear()


    def draw_points_on_image(self, pixmap):
        draw_pixmap = pixmap.copy()
        painter = QPainter(draw_pixmap)
        
        pen = QPen(Qt.red, 8, Qt.SolidLine)
        painter.setPen(pen)
        
        painter.drawEllipse(self.origin[0], self.origin[1], 8, 8)
        painter.drawText(self.origin[0] + 10, self.origin[1] - 10, '0')

        for i, (x, y) in enumerate(self.points + self.points_detection):
            # dist_x = abs(self.meter_to_pixel(x))
            # dist_y = abs(self.meter_to_pixel(y)
            
            # x = self.origin[0] + dist_x
            # y = self.origin[1] + dist_y

            point, _ = cv2.projectPoints(np.array([[x,y,0]], dtype=np.float32), self.rvec, self.tvec, self.camera_matrix, self.distortion)
            x, y = (int(point[0][0][0]), int(point[0][0][1]))
            if self.show_cropped:
                point = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), self.H)[0][0]
                x, y = (int(point[0]), int(point[1]))


            painter.drawEllipse(x, y, 4, 4)
            painter.drawText(x + 10, y - 10, str(i + 1))
            
        painter.end()
        return draw_pixmap
    
    def img_to_realworld(self, x, y):
        Lcam = self.camera_matrix.dot(np.hstack((cv2.Rodrigues(self.rvec)[0],self.tvec)))
        if self.show_cropped:
            # Adjust coordinates based on current frame size
            x, y = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), np.linalg.inv(self.H))[0][0]
        Z=0
        point = np.linalg.inv(np.hstack((Lcam[:,0:2],np.array([[-1*x],[-1*y],[-1]])))).dot((-Z*Lcam[:,2]-Lcam[:,3]))
        return point[0], point[1]

        
    def on_point_clicked(self, x, y, ):
        """Handle point selection on interactive image"""
        if self.current_frame is not None:
            real_x, real_y = self.img_to_realworld(x, y)

            dist = np.sqrt(real_x**2 + real_y**2)
            self.points.append((
                real_x,
                real_y
            ))
            dist_text = f"Distance between point {len(self.points)} and origin: {dist:.2f} pixels, or {self.pixel_to_meter(dist):.2f} mm\n"
            dist_text += f"x: {self.points[-1][0]:.2f} mm, y: {self.points[-1][1]:.2f} mm\n---------------------------------"
            self.points_list.addItem(dist_text)

                
    def clear_points(self):
        """Clear all selected points"""
        self.points.clear()
        self.points_list.clear()
    
    def toggle_picking(self, checked):
        self.is_picking_enabled = checked
        if checked:
            self.move_robot()
    
    def move_robot(self):
        if len(self.points_detection) == 0:
            QMessageBox.warning(self, "No Points", "No points selected for picking.")
            return

        self.stop_camera()
        
        start_pose = self.CONFIGS['jaka']['start_pose']
        speed = self.CONFIGS['jaka']['speed']
        coordinate_system = self.CONFIGS['jaka']['coordinate_system_index']
        tcp = self.CONFIGS['jaka']['tcp_index']
        rot_x = self.CONFIGS['jaka']['tcp_rotation_xy'][0]
        rot_y = self.CONFIGS['jaka']['tcp_rotation_xy'][1]
        object_height = self.CONFIGS['jaka']['object_height']
        pick_force = self.CONFIGS['jaka']['pick_force']
        digital_output_index = self.CONFIGS['jaka']['digital_output_index']
        drop_position = self.CONFIGS['jaka']['drop_position']
        
        robot = RC(
            self.CONFIGS['jaka']['ip_address']
        )
        robot.login()
        robot.power_on()
        robot.enable_robot()
        print("Robot connected and enabled.")
        robot.set_user_frame_id(coordinate_system)
        robot.set_tool_id(tcp)
        robot.set_torsenosr_brand(10)   # 10 means embedded force sensor
        # robot.set_ft_ctrl_frame(1)      # 0 -> tool, 1 -> world
        robot.set_torque_sensor_mode(1) # 1 -> on, 0 -> off
        print("User coordinate system: " + str(robot.get_user_frame_id()[1]))
        print("TCP setting: " + str(robot.get_tool_id()[1]))
        # print("Force sensor index:" + str(robot.get_torsenosr_brand()[1]))              # 10 means embedded force sensor
        # print("Force sensor mode: " + str(robot.get_compliant_type()))                  # [0, sensor_compensation, force_control_type]
        # print("Force sensor coordinate_system: " + str(robot.get_ft_ctrl_frame()[1]))   # 0 -> tool, 1 -> world
        # print("Payload: " + str(robot.get_torq_sensor_tool_payload()[1]))               # kg, [x, y, z]
        # print("Current torque sensor data: " + str(robot.get_torque_sensor_data(2)[1][2])) # index from 0 to 4 maybe better 1?

        # robot.set_admit_ctrl_config(0, 0, 0, 0, 0, 0)
        # robot.set_admit_ctrl_config(1, 0, 0, 0, 0, 0)
        # robot.set_admit_ctrl_config(2, 1, 10, -20, -10, -10)
        # robot.set_admit_ctrl_config(3, 0, 0, 0, 0, 0)
        # robot.set_admit_ctrl_config(4, 0, 0, 0, 0, 0)
        # robot.set_admit_ctrl_config(5, 0, 0, 0, 0, 0)
        # robot.enable_admittance_ctrl(1)

        ############### CONTROLLO IN FORZA QUI ##################
        # ret = z
        # max_force = 0
        # while True:
        #     force = robot.get_torque_sensor_data(1)[1][2][2]
        #     if abs(force) > max_force:
        #         max_force = abs(force)
        #     if max_force > 1:
        #         robot.motion_abort()
        #         break
        #     print("Current torque sensor data: " + str(robot.get_torque_sensor_data(1)[1][2])) # index from 0 to 4 maybe better 1?
        #     #time.sleep(0.1)
        # print("Max force during homing: ", max_force)
        # print(ret)
        # print("Homing done.")
        
        # Move to charuko origin
        # res = robot.linear_move([0,0,20,rot_x, rot_y,3.14], 0, True, speed)[0]
        # print("Moved to start pose." + str(res))
        # if res != 0 and res != -12:
        #     print("Failed to move to start pose, aborting.")
        #     robot.logout()
        #     QMessageBox.warning(self, "Robot error", "Linear move errro" + str(res))
        #     return
        # return
        

        Z = object_height
        for x,y in self.points_detection:
            target_pos = [
                y,
                x,
                Z,
                rot_x,
                rot_y,
                3.14
            ]
            print(f"Moving to point: x={x} mm, y={y} mm")
            if not self.is_picking_enabled:
                break
            res = robot.linear_move(target_pos, 0, True, speed)


            max_force = 0
            offset_z = object_height + 3
            robot.zero_end_sensor()
            # Move down to pick
            robot.linear_move([0,0, -offset_z, 0, 0, 0], 1, False, 20)
            start_time = time.time()
            while True:
                force = abs(robot.get_torque_sensor_data(1)[1][2][2])
                if force > max_force:
                    max_force = force
                if max_force >= pick_force or not self.is_picking_enabled:
                    robot.motion_abort()
                    break
                current_time = time.time()
                if current_time - start_time > 5:
                    robot.linear_move([0,0, -offset_z, 0, 0, 0], 1, False, 20)
                    print("Timeout reached while picking, aborting.")
                    start_time = current_time
                
            if not self.is_picking_enabled:
                break
            print("Picked object with force: ", max_force)
            robot.set_digital_output(0, digital_output_index, 1)

            robot.linear_move([0,0, offset_z, 0, 0, 0], 1, True, 20)
            robot.linear_move(drop_position, 0, True, speed)

            robot.set_digital_output(0, digital_output_index, 0)

            time.sleep(1.5)

        robot.motion_abort()
        robot.logout()



def main():
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()