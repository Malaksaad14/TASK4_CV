import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QComboBox, QSlider, QGroupBox, QFormLayout, QSpinBox)
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

class ThresholdingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Image Thresholding - Part A")
        self.resize(1200, 800)
        
        self.original_image = None
        self.processed_image = None
        self.current_image_array = None
        
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        sidebar_layout = QVBoxLayout(sidebar)
        
        # Image Loading
        load_group = QGroupBox("Input")
        load_layout = QVBoxLayout()
        self.btn_load = QPushButton("Open Image")
        self.btn_load.clicked.connect(self.load_image)
        load_layout.addWidget(self.btn_load)
        load_group.setLayout(load_layout)
        sidebar_layout.addWidget(load_group)

        # Method Selection
        method_group = QGroupBox("Thresholding Method")
        method_layout = QVBoxLayout()
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Optimal Thresholding", "Otsu Thresholding", 
                                   "Spectral (Multilevel)", "Local Thresholding"])
        self.combo_method.currentIndexChanged.connect(self.update_params_visibility)
        method_layout.addWidget(self.combo_method)
        method_group.setLayout(method_layout)
        sidebar_layout.addWidget(method_group)

        # Parameters
        self.params_group = QGroupBox("Parameters")
        self.params_layout = QFormLayout()
        
        # Local Thresholding Params
        self.spin_block = QSpinBox()
        self.spin_block.setRange(3, 101)
        self.spin_block.setSingleStep(2)
        self.spin_block.setValue(35)
        self.spin_block.valueChanged.connect(self.apply_thresholding) # Connect for real-time update
        self.params_layout.addRow("Block Size:", self.spin_block)
        
        self.spin_offset = QSpinBox()
        self.spin_offset.setRange(-50, 50)
        self.spin_offset.setValue(10)
        self.spin_offset.valueChanged.connect(self.apply_thresholding) # Connect for real-time update
        self.params_layout.addRow("Offset:", self.spin_offset)
        
        # Spectral Params
        self.spin_modes = QSpinBox()
        self.spin_modes.setRange(2, 5)
        self.spin_modes.setValue(3)
        self.spin_modes.valueChanged.connect(self.apply_thresholding) # Connect for real-time update
        self.params_layout.addRow("Modes:", self.spin_modes)
        
        self.params_group.setLayout(self.params_layout)
        sidebar_layout.addWidget(self.params_group)
        
        sidebar_layout.addStretch()
        
        self.btn_apply = QPushButton("Apply Thresholding")
        self.btn_apply.setFixedHeight(50)
        self.btn_apply.clicked.connect(self.apply_thresholding)
        sidebar_layout.addWidget(self.btn_apply)
        
        main_layout.addWidget(sidebar)

        # Image Display Area
        display_layout = QHBoxLayout()
        
        self.lbl_orig = QLabel("Original Image")
        self.lbl_orig.setAlignment(Qt.AlignCenter)
        self.lbl_orig.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")
        
        self.lbl_proc = QLabel("Processed Image")
        self.lbl_proc.setAlignment(Qt.AlignCenter)
        self.lbl_proc.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")
        
        display_layout.addWidget(self.lbl_orig)
        display_layout.addWidget(self.lbl_proc)
        
        main_layout.addLayout(display_layout)
        
        self.update_params_visibility()

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #ffffff; }
            QPushButton { 
                background-color: #2c3e50; color: white; 
                border-radius: 5px; padding: 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #34495e; }
            QGroupBox { font-weight: bold; margin-top: 10px; }
            QLabel { color: #2c3e50; }
        """)

    def update_params_visibility(self):
        method = self.combo_method.currentText()
        is_local = method == "Local Thresholding"
        is_spectral = method == "Spectral (Multilevel)"
        
        self.spin_block.setEnabled(is_local)
        self.spin_offset.setEnabled(is_local)
        self.spin_modes.setEnabled(is_spectral)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "images", "Image files (*.png *.jpg *.jpeg *.bmp)")
        if path:
            img = Image.open(path).convert('L')
            self.current_image_array = np.array(img)
            self.display_image(self.current_image_array, self.lbl_orig)
            self.lbl_proc.clear()
            self.lbl_proc.setText("Click 'Apply' to see results")

    def display_image(self, arr, label):
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def apply_thresholding(self):
        if self.current_image_array is None:
            return
            
        method = self.combo_method.currentText()
        arr = self.current_image_array
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if method == "Optimal Thresholding":
                t = optimal_thresholding(arr)
                res = apply_threshold(arr, t)
            elif method == "Otsu Thresholding":
                t = otsu_thresholding(arr)
                res = apply_threshold(arr, t)
            elif method == "Spectral (Multilevel)":
                # Currently implemented for 2 thresholds (3 modes)
                t = spectral_thresholding(arr, num_thresholds=2)
                res = apply_multilevel_threshold(arr, t)
            elif method == "Local Thresholding":
                bs = self.spin_block.value()
                off = self.spin_offset.value()
                res = local_thresholding(arr, block_size=bs, offset=off)
            
            self.display_image(res, self.lbl_proc)
        finally:
            QApplication.restoreOverrideCursor()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ThresholdingGUI()
    gui.show()
    sys.exit(app.exec_())
