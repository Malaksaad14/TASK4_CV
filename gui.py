import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QComboBox, QSlider, QGroupBox, QFormLayout, QSpinBox,
                             QTabWidget)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from PIL import Image

from thresholding import (
    optimal_thresholding,
    otsu_thresholding,
    spectral_thresholding,
    local_thresholding,
    apply_threshold,
    apply_multilevel_threshold
)

from segmentation import (
    kmeans_segmentation,
    mean_shift_segmentation,
    region_growing_segmentation,
    agglomerative_segmentation
)

class ImageProcessingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Image Processing - Project 4")
        self.resize(1200, 800)
        
        self.original_image = None
        self.processed_image = None
        
        # We store both grayscale and rgb for thresholding vs segmentation
        self.current_image_gray = None
        self.current_image_color = None
        
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Tabs for Sidebar
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(350)
        
        # Tab 1: Thresholding
        self.tab_thresh = QWidget()
        self.init_thresholding_tab()
        self.tabs.addTab(self.tab_thresh, "Part A: Thresholding")
        
        # Tab 2: Segmentation
        self.tab_seg = QWidget()
        self.init_segmentation_tab()
        self.tabs.addTab(self.tab_seg, "Part B: Segmentation")
        
        main_layout.addWidget(self.tabs)

        # Image Display Area
        display_layout = QHBoxLayout()
        
        self.lbl_orig = QLabel("Original Image")
        self.lbl_orig.setAlignment(Qt.AlignCenter)
        self.lbl_orig.setStyleSheet("border: 2px dashed #45475a; background: #11111b; border-radius: 10px;")
        
        self.lbl_proc = QLabel("Processed Image")
        self.lbl_proc.setAlignment(Qt.AlignCenter)
        self.lbl_proc.setStyleSheet("border: 2px dashed #45475a; background: #11111b; border-radius: 10px;")
        
        display_layout.addWidget(self.lbl_orig)
        display_layout.addWidget(self.lbl_proc)
        
        # Create a container for display to add the load button at the top
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        
        load_group = QGroupBox("Global Controls")
        load_layout = QHBoxLayout()
        self.btn_load = QPushButton("Open Image")
        self.btn_load.clicked.connect(self.load_image)
        load_layout.addWidget(self.btn_load)
        load_group.setLayout(load_layout)
        
        right_layout.addWidget(load_group)
        right_layout.addLayout(display_layout)
        
        main_layout.addWidget(right_container)
        
        self.update_params_visibility_thresh()
        self.update_params_visibility_seg()

    def init_thresholding_tab(self):
        layout = QVBoxLayout(self.tab_thresh)
        
        # Method Selection
        method_group = QGroupBox("Thresholding Method")
        method_layout = QVBoxLayout()
        self.combo_method_thresh = QComboBox()
        self.combo_method_thresh.addItems(["Optimal Thresholding", "Otsu Thresholding", 
                                   "Spectral (Multilevel)", "Local Thresholding"])
        self.combo_method_thresh.currentIndexChanged.connect(self.update_params_visibility_thresh)
        method_layout.addWidget(self.combo_method_thresh)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Parameters
        self.params_group_thresh = QGroupBox("Parameters")
        self.params_layout_thresh = QFormLayout()
        
        # Local Thresholding Params
        self.spin_block = QSpinBox()
        self.spin_block.setRange(3, 101)
        self.spin_block.setSingleStep(2)
        self.spin_block.setValue(35)
        self.params_layout_thresh.addRow("Block Size:", self.spin_block)
        
        self.spin_offset = QSpinBox()
        self.spin_offset.setRange(-50, 50)
        self.spin_offset.setValue(10)
        self.params_layout_thresh.addRow("Offset:", self.spin_offset)
        
        # Spectral Params
        self.spin_modes = QSpinBox()
        self.spin_modes.setRange(2, 5)
        self.spin_modes.setValue(3)
        self.params_layout_thresh.addRow("Modes:", self.spin_modes)
        
        self.params_group_thresh.setLayout(self.params_layout_thresh)
        layout.addWidget(self.params_group_thresh)
        
        layout.addStretch()
        
        self.btn_apply_thresh = QPushButton("Apply Thresholding")
        self.btn_apply_thresh.setFixedHeight(50)
        self.btn_apply_thresh.clicked.connect(self.apply_thresholding)
        layout.addWidget(self.btn_apply_thresh)

    def init_segmentation_tab(self):
        layout = QVBoxLayout(self.tab_seg)
        
        # Method Selection
        method_group = QGroupBox("Segmentation Method")
        method_layout = QVBoxLayout()
        self.combo_method_seg = QComboBox()
        self.combo_method_seg.addItems(["K-Means", "Region Growing", 
                                   "Agglomerative", "Mean Shift"])
        self.combo_method_seg.currentIndexChanged.connect(self.update_params_visibility_seg)
        method_layout.addWidget(self.combo_method_seg)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Parameters
        self.params_group_seg = QGroupBox("Parameters")
        self.params_layout_seg = QFormLayout()
        
        # K-Means & Agglomerative Params
        self.spin_clusters = QSpinBox()
        self.spin_clusters.setRange(2, 20)
        self.spin_clusters.setValue(3)
        self.params_layout_seg.addRow("Clusters (K):", self.spin_clusters)
        
        # Region Growing Params
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(1, 100)
        self.spin_threshold.setValue(20)
        self.params_layout_seg.addRow("Tolerance:", self.spin_threshold)
        
        # Mean Shift Params
        self.spin_sp_radius = QSpinBox()
        self.spin_sp_radius.setRange(1, 100)
        self.spin_sp_radius.setValue(20)
        self.params_layout_seg.addRow("Spatial Radius:", self.spin_sp_radius)
        
        self.spin_col_radius = QSpinBox()
        self.spin_col_radius.setRange(1, 100)
        self.spin_col_radius.setValue(40)
        self.params_layout_seg.addRow("Color Radius:", self.spin_col_radius)
        
        self.params_group_seg.setLayout(self.params_layout_seg)
        layout.addWidget(self.params_group_seg)
        
        layout.addStretch()
        
        self.btn_apply_seg = QPushButton("Apply Segmentation")
        self.btn_apply_seg.setFixedHeight(50)
        self.btn_apply_seg.clicked.connect(self.apply_segmentation)
        layout.addWidget(self.btn_apply_seg)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #1e1e2e; 
            }
            QWidget {
                color: #cdd6f4;
                font-family: 'Segoe UI', Inter, sans-serif;
                font-size: 14px;
            }
            QTabWidget::pane {
                border: 1px solid #313244;
                border-radius: 8px;
                background-color: #181825;
            }
            QTabBar::tab {
                background: #313244;
                color: #a6adc8;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #cba6f7;
                color: #11111b;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: #45475a;
            }
            QPushButton { 
                background-color: #89b4fa; 
                color: #11111b; 
                border-radius: 6px; 
                padding: 12px; 
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { 
                background-color: #b4befe; 
            }
            QPushButton:pressed {
                background-color: #74c7ec;
            }
            QGroupBox { 
                font-weight: bold; 
                border: 1px solid #45475a; 
                border-radius: 8px; 
                margin-top: 20px;
                padding-top: 15px;
                color: #f38ba8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
            }
            QLabel { 
                color: #cdd6f4; 
            }
            QComboBox, QSpinBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox:hover, QSpinBox:hover {
                border: 1px solid #89b4fa;
            }
        """)

    def update_params_visibility_thresh(self):
        method = self.combo_method_thresh.currentText()
        is_local = method == "Local Thresholding"
        is_spectral = method == "Spectral (Multilevel)"
        
        self.spin_block.setEnabled(is_local)
        self.spin_offset.setEnabled(is_local)
        self.spin_modes.setEnabled(is_spectral)

    def update_params_visibility_seg(self):
        method = self.combo_method_seg.currentText()
        
        self.spin_clusters.setVisible(method in ["K-Means", "Agglomerative"])
        self.spin_threshold.setVisible(method == "Region Growing")
        self.spin_sp_radius.setVisible(method == "Mean Shift")
        self.spin_col_radius.setVisible(method == "Mean Shift")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "images", "Image files (*.png *.jpg *.jpeg *.bmp)")
        if path:
            img_color = Image.open(path).convert('RGB')
            img_gray = Image.open(path).convert('L')
            
            self.current_image_color = np.array(img_color)
            self.current_image_gray = np.array(img_gray)
            
            self.display_image(self.current_image_color, self.lbl_orig, is_color=True)
            self.lbl_proc.clear()
            self.lbl_proc.setText("Click 'Apply' to see results")

    def display_image(self, arr, label, is_color=False):
        if arr is None: return
        h, w = arr.shape[:2]
        
        if is_color or len(arr.shape) == 3:
            # OpenCV image format or PIL -> PyQt needs RGB
            if arr.shape[2] == 3:
                qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
            else:
                qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        else:
            qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
            
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def apply_thresholding(self):
        if self.current_image_gray is None:
            return
            
        method = self.combo_method_thresh.currentText()
        arr = self.current_image_gray.copy()
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if method == "Optimal Thresholding":
                t = optimal_thresholding(arr)
                res = apply_threshold(arr, t)
            elif method == "Otsu Thresholding":
                t = otsu_thresholding(arr)
                res = apply_threshold(arr, t)
            elif method == "Spectral (Multilevel)":
                t = spectral_thresholding(arr, num_thresholds=2)
                res = apply_multilevel_threshold(arr, t)
            elif method == "Local Thresholding":
                bs = self.spin_block.value()
                off = self.spin_offset.value()
                res = local_thresholding(arr, block_size=bs, offset=off)
            
            self.display_image(res, self.lbl_proc, is_color=False)
        finally:
            QApplication.restoreOverrideCursor()

    def apply_segmentation(self):
        if self.current_image_color is None:
            return
            
        method = self.combo_method_seg.currentText()
        arr = self.current_image_color.copy()
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if method == "K-Means":
                k = self.spin_clusters.value()
                res = kmeans_segmentation(arr, n_clusters=k)
            elif method == "Region Growing":
                thresh = self.spin_threshold.value()
                res = region_growing_segmentation(arr, threshold=thresh)
            elif method == "Agglomerative":
                k = self.spin_clusters.value()
                res = agglomerative_segmentation(arr, n_clusters=k)
            elif method == "Mean Shift":
                sp = self.spin_sp_radius.value()
                col = self.spin_col_radius.value()
                res = mean_shift_segmentation(arr, spatial_radius=sp, color_radius=col)
                
            self.display_image(res, self.lbl_proc, is_color=True)
        finally:
            QApplication.restoreOverrideCursor()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ImageProcessingGUI()
    gui.show()
    sys.exit(app.exec_())
