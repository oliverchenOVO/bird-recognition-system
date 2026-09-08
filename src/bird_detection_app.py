#!/usr/bin/env python3
"""
桌面鸟类识别应用程序
结合训练好的YOLO模型和Web界面，创建独立的桌面应用
"""

import sys
import os
import json
import threading
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from PyQt6.QtGui import QPixmap, QImage

# PyQt6 用于桌面应用
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                            QHBoxLayout, QWidget, QPushButton, QLabel, 
                            QFileDialog, QSlider, QTableWidget, QTableWidgetItem,
                            QTextEdit, QSplitter, QGroupBox, QProgressBar,
                            QMessageBox, QTabWidget, QScrollArea, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QIcon
from PyQt6.QtWebEngineWidgets import QWebEngineView

# YOLO 和相关库
from ultralytics import YOLO
import torch

class YOLODetector:
    """YOLO检测器类"""

    def __init__(self, model_path=None):
        self.model = None
        self.class_names = []
        self.model_loaded = False

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, file_path=None):
        """加载YOLO模型"""
        if file_path is None:
            file_path, _ = QFileDialog.getOpenFileName(
                None,
                "选择YOLO模型文件",
                "",
                "模型文件 (*.pt *.onnx);;所有文件 (*)"
            )
        if not file_path:
            return False

        try:
            self.model = YOLO(file_path)
            self.class_names = list(self.model.names.values())
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None
            self.class_names = []
            self.model_loaded = False
            return False
        
    
    def detect(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """执行检测"""
        if not self.model_loaded:
            return []
        
        try:
            # 执行推理
            results = self.model(image_path, conf=conf_threshold, iou=iou_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    # 提取检测结果
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy()
                    
                    for i, (box, confidence, class_id) in enumerate(zip(xyxy, conf, cls)):
                        class_name = self.class_names[int(class_id)]
                        detections.append({
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': [float(x) for x in box],  # [x1, y1, x2, y2]
                            'class_id': int(class_id)
                        })
            
            return detections
            
        except Exception as e:
            print(f"检测过程出错: {e}")
            return []


class DetectionThread(QThread):
    """检测线程"""
    
    detection_finished = pyqtSignal(list)
    detection_progress = pyqtSignal(str)
    
    def __init__(self, detector, image_path, conf_threshold, iou_threshold):
        super().__init__()
        self.detector = detector
        self.image_path = image_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def run(self):
        """执行检测"""
        self.detection_progress.emit("正在进行鸟类检测...")
        
        # 执行检测
        detections = self.detector.detect(
            self.image_path, 
            self.conf_threshold, 
            self.iou_threshold
        )
        
        self.detection_progress.emit(f"检测完成，发现 {len(detections)} 只鸟类")
        self.detection_finished.emit(detections)


class ImageDisplayWidget(QLabel):
    """图片显示组件"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                background-color: #f9f9f9;
            }
        """)
        self.setText("请选择图片文件\n支持格式: JPG, PNG, BMP")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        
        self.original_pixmap = None
        self.detections = []
        
    def load_image(self, image_path):
        """加载图片"""
        try:
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                return False
            
            # 缩放图片以适应显示区域
            scaled_pixmap = pixmap.scaled(
                self.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.original_pixmap = pixmap
            self.setPixmap(scaled_pixmap)
            return True
            
        except Exception as e:
            print(f"图片加载失败: {e}")
            return False
    
    def draw_detections(self, detections, class_name_map=None):
        """在图片上绘制检测结果"""
        if not self.original_pixmap or not detections:
            return

        self.detections = detections

        # 创建可绘制的图片副本
        pixmap = self.original_pixmap.copy()
        painter = QPainter(pixmap)

        # 设置绘制参数
        pen = QPen(QColor(255, 68, 68), 3)  # 红色边框
        painter.setPen(pen)
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))

        # 计算缩放比例
        img_width = pixmap.width()
        img_height = pixmap.height()

        for detection in detections:
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            class_name = detection['class']
            confidence = detection['confidence']

            # 获取中文名
            if '.' in class_name:
                class_en_key = class_name.split('.', 1)[1]
            else:
                class_en_key = class_name
            # 优先用传入的映射表
            if class_name_map:
                class_cn = class_name_map.get(class_en_key, class_en_key)
            # 否则尝试用父窗口的
            elif hasattr(self.parent(), 'class_name_map'):
                class_cn = self.parent().class_name_map.get(class_en_key, class_en_key)
            else:
                class_cn = class_en_key

            # 绘制边界框
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            painter.drawRect(x1, y1, x2-x1, y2-y1)

            # 只显示中文+置信度
            label = f"{class_cn} {confidence:.2f}"

            # 标签背景
            label_rect = painter.fontMetrics().boundingRect(label)
            label_rect.adjust(-5, -5, 5, 5)
            label_rect.moveTo(x1, y1 - label_rect.height())

            painter.fillRect(label_rect, QColor(255, 68, 68))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)
            painter.setPen(pen)

        painter.end()

        # 缩放并显示
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.setPixmap(scaled_pixmap)

    
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放图片"""
        super().resizeEvent(event)
        if self.original_pixmap:
            self.draw_detections(self.detections)


class HeatmapWidget(QLabel):
    """热力图显示组件"""
    
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 200)
        self.setStyleSheet("""
            QLabel {
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #f5f5f5;
            }
        """)
        self.setText("等待检测结果")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def generate_heatmap(self, detections, image_width, image_height):
        """生成热力图"""
        if not detections:
            return
        
        # 创建热力图
        heatmap = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # 创建渐变背景
        for i in range(200):
            for j in range(300):
                # 径向渐变
                center_x, center_y = 150, 100
                distance = np.sqrt((j - center_x)**2 + (i - center_y)**2)
                intensity = max(0, 1 - distance / 100)
                
                heatmap[i, j] = [
                    int(255 * intensity * 0.2),  # R
                    int(255 * intensity * 0.5),  # G  
                    int(255 * intensity * 0.8)   # B
                ]
        
        # 在检测位置添加热点
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # 转换坐标到热力图尺寸
            center_x = int((bbox[0] + bbox[2]) / 2 / image_width * 300)
            center_y = int((bbox[1] + bbox[3]) / 2 / image_height * 200)
            
            # 添加高亮热点
            radius = int(30 * confidence)
            y, x = np.ogrid[:200, :300]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            if np.any(mask):
                # 获取需要修改的像素位置
                masked_pixels = heatmap[mask]
                
                # 安全地更新像素值
                new_r = np.clip(masked_pixels[:, 0] + int(255 * confidence), 0, 255)
                new_g = np.clip(masked_pixels[:, 1] + int(128 * confidence), 0, 255)
                new_b = masked_pixels[:, 2]  # 保持蓝色通道不变
                
                # 更新热力图
                heatmap[mask, 0] = new_r
                heatmap[mask, 1] = new_g
                heatmap[mask, 2] = new_b
        
        # 转换为QImage再转为QPixmap并显示
        height, width, channel = heatmap.shape
        bytes_per_line = 3 * width
        q_image = QImage(heatmap.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_image))


class BirdDetectionApp(QMainWindow):
    """主应用程序窗口"""

    def __init__(self):
        super().__init__()
        self.detector = YOLODetector()
        self.current_image_path = None
        self.current_detections = []

        # 自动加载英文-中文类别映射表
        self.class_name_map = self.load_class_name_map()

        self.init_ui()
        self.setup_connections()

    def load_class_name_map(self):
        """从 classes.txt 读取英文-中文类别映射表"""
        class_name_map = {}
        classes_txt = Path(__file__).resolve().parents[1] / "data" / "classes.txt"
        # 你可以根据实际情况修改为你的中文翻译列表
        chinese_names = [
            "黑脚信天翁", "莱桑信天翁", "黑褐信天翁", "有槽嘴的阿尼鸟", "冠海雀", "小海雀", "鹦鹉海雀", "犀鸟海雀", "布鲁尔黑鸟", "红翅黑鸟",
            "锈黑鸟", "黄头黑鸟", "稻鹀", "靛蓝彩鹀", "青蓝彩鹀", "彩绘彩鹀", "红衣主教", "斑猫鸟", "灰猫鸟", "黄胸鹀",
            "东拖鹀", "查克威尔寡妇鸟", "布兰特鸬鹚", "红脸鸬鹚", "海鸬鹚", "青铜牛鸟", "闪亮牛鸟", "褐爬树鸟", "美洲鸦", "鱼鸦",
            "黑嘴杜鹃", "红树林杜鹃", "黄嘴杜鹃", "普通黄喉鹀", "紫雀", "北啄木鸟", "阿卡迪安蝇鹟", "大冠蝇鹟", "最小蝇鹟", "橄榄侧蝇鹟",
            "剪尾蝇鹟", "朱红蝇鹟", "黄腹蝇鹟", "军舰鸟", "北暴风鹱", "赤膀鸭", "美洲金翅雀", "欧洲金翅雀", "船尾咕咕鸟", "耳鸊鷉",
            "角鸊鷉", "厚嘴鸊鷉", "西方鸊鷉", "蓝彩鹀", "晚彩鹀", "松彩鹀", "玫瑰胸彩鹀", "鸽海雀", "加州鸥", "灰翅鸥",
            "赫尔曼鸥", "银鸥", "象牙鸥", "环嘴鸥", "板背鸥", "西方鸥", "安娜蜂鸟", "红喉蜂鸟", "红褐蜂鸟", "绿紫耳蜂鸟",
            "长尾贼鸥", "庞马林贼鸥", "蓝松鸦", "佛罗里达松鸦", "绿松鸦", "暗眼灯鹀", "热带王鹟", "灰王鹟", "带带翠鸟", "绿翠鸟",
            "斑翠鸟", "环翠鸟", "白胸翠鸟", "红腿三趾鸥", "角云雀", "太平洋潜鸟", "绿头鸭", "西方草地鹨", "兜帽秋沙鸭", "红胸秋沙鸭",
            "嘲鸫", "夜鹰", "克拉克坚果鸟", "白胸鹪鹩", "巴尔的摩拟鹂", "兜帽拟鹂", "果园拟鹂", "斯科特拟鹂", "灶鸟", "褐鹈鹕",
            "白鹈鹕", "西方林鹟", "赛约尼斯", "美洲鹨", "夜鹰", "角嘴海雀", "普通渡鸦", "白颈渡鸦", "美洲红尾鸲", "地咬鹃",
            "灰背伯劳", "大灰伯劳", "贝尔德麻雀", "黑喉麻雀", "布鲁尔麻雀", "切皮麻雀", "黏土色麻雀", "家麻雀", "田麻雀", "狐麻雀",
            "蚱蜢麻雀", "哈里斯麻雀", "亨斯洛麻雀", "勒孔特麻雀", "林肯麻雀", "纳尔逊锐尾麻雀", "草地麻雀", "海滨麻雀", "歌麻雀", "树麻雀",
            "田园麻雀", "白冠麻雀", "白喉麻雀", "光泽椋鸟", "岸燕", "谷燕", "崖燕", "树燕", "猩红唐纳雀", "夏唐纳雀",
            "北极燕鸥", "黑燕鸥", "里海燕鸥", "普通燕鸥", "优雅燕鸥", "福斯特燕鸥", "最小燕鸥", "绿尾拖鹀", "褐色画眉", "鼠尾画眉",
            "黑顶绿鹃", "蓝头绿鹃", "费城绿鹃", "红眼绿鹃", "歌唱绿鹃", "白眼绿鹃", "黄喉绿鹃", "栗胸莺", "黑白莺", "黑喉蓝莺",
            "蓝翅莺", "加拿大莺", "五月莺", "天蓝莺", "栗胸侧莺", "金翅莺", "兜帽莺", "肯塔基莺", "玉兰莺", "哀莺",
            "桃金娘莺", "松莺", "草原莺", "金莺", "斯温森莺", "田纳西莺", "威尔逊莺", "蠕虫食莺", "黄莺", "北水鹨",
            "路易斯安那水鹨", "波希米亚太平鸟", "雪松太平鸟", "美洲三趾啄木鸟", "冠啄木鸟", "红腹啄木鸟", "红冠啄木鸟", "红头啄木鸟", "斑啄木鸟", "比威克鹪鹩",
            "仙人掌鹪鹩", "卡罗莱纳鹪鹩", "家鹪鹩", "沼泽鹪鹩", "岩鹪鹩", "冬鹪鹩", "普通黄喉鹀"
        ]
        try:
            with open(classes_txt, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        class_id, class_en = parts
                        # 去掉前缀编号
                        class_en_name = class_en.split('.', 1)[1] if '.' in class_en else class_en
                        # 中文名按顺序匹配
                        class_cn = chinese_names[idx] if idx < len(chinese_names) else class_en_name
                        class_name_map[class_en_name] = class_cn
        except Exception as e:
            print(f"加载类别映射表失败: {e}")
        return class_name_map
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("基于YOLOv8的鸟类识别系统")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置应用图标（可选）
        # self.setWindowIcon(QIcon("icon.png"))
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧面板 - 图片显示和控制
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # 右侧面板 - 信息显示
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # 底部面板 - 结果表格
        bottom_panel = self.create_bottom_panel()
        
        # 整体布局调整
        main_container = QWidget()
        container_layout = QVBoxLayout(main_container)
        container_layout.addWidget(splitter, 3)  # 上部分占3/4
        container_layout.addWidget(bottom_panel, 1)  # 下部分占1/4
        
        self.setCentralWidget(main_container)
        
        # 设置分割器比例
        splitter.setSizes([800, 400])
        
    def create_left_panel(self):
        """创建左侧面板"""
        panel = QGroupBox("图像检测")
        layout = QVBoxLayout(panel)
        
        # 控制按钮区域
        controls_layout = QHBoxLayout()
        
        # 模型加载按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        controls_layout.addWidget(self.load_model_btn)
        
        # 图片选择按钮
        self.select_image_btn = QPushButton("选择图片")
        self.select_image_btn.clicked.connect(self.select_image)
        controls_layout.addWidget(self.select_image_btn)
        
        # 检测按钮
        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.clicked.connect(self.start_detection)
        self.detect_btn.setEnabled(False)
        controls_layout.addWidget(self.detect_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # 参数调节区域
        params_layout = QHBoxLayout()
        
        # 置信度滑块
        params_layout.addWidget(QLabel("CONF:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)  # 0.25
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        params_layout.addWidget(self.conf_slider)
        
        self.conf_label = QLabel("0.25")
        params_layout.addWidget(self.conf_label)
        
        # IOU滑块
        params_layout.addWidget(QLabel("IOU:"))
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)  # 0.45
        self.iou_slider.valueChanged.connect(self.on_iou_changed)
        params_layout.addWidget(self.iou_slider)
        
        self.iou_label = QLabel("0.45")
        params_layout.addWidget(self.iou_label)
        
        # 检测数量显示
        params_layout.addWidget(QLabel("目标数目:"))
        self.target_count_label = QLabel("0")
        self.target_count_label.setStyleSheet("""
            QLabel {
                background-color: #4CAF50;
                color: white;
                padding: 5px 10px;
                border-radius: 15px;
                font-weight: bold;
            }
        """)
        params_layout.addWidget(self.target_count_label)
        
        params_layout.addStretch()
        layout.addLayout(params_layout)
        
        # 图片显示区域
        self.image_display = ImageDisplayWidget()
        layout.addWidget(self.image_display)
        
        return panel
        
    def create_right_panel(self):
        """创建右侧面板"""
        panel = QGroupBox("检测信息")
        layout = QVBoxLayout(panel)
        
        # 状态信息
        status_group = QGroupBox("状态信息")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("等待加载模型...")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #e8f5e8;
                border: 1px solid #4CAF50;
                border-radius: 8px;
                padding: 10px;
                color: #2e7d32;
            }
        """)
        status_layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        layout.addWidget(status_group)
        
        # 热力图
        heatmap_group = QGroupBox("热力图")
        heatmap_layout = QVBoxLayout(heatmap_group)
        
        self.heatmap_display = HeatmapWidget()
        heatmap_layout.addWidget(self.heatmap_display)
        
        layout.addWidget(heatmap_group)
        
        # 支持的鸟类列表
        classes_group = QGroupBox("支持的鸟类")
        classes_layout = QVBoxLayout(classes_group)
        
        self.classes_text = QTextEdit()
        self.classes_text.setMaximumHeight(150)
        self.classes_text.setReadOnly(True)
        self.classes_text.setText("请先加载模型...")
        classes_layout.addWidget(self.classes_text)
        
        layout.addWidget(classes_group)
        
        layout.addStretch()
        return panel
    
    def create_bottom_panel(self):
        """创建底部结果表格面板"""
        panel = QGroupBox("检测结果记录")
        layout = QVBoxLayout(panel)
        
        # 创建表格
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "序号", "图像信息", "结果", "位置", "置信度"
        ])
        
        # 设置表格样式
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        # 初始化空表格
        self.results_table.setRowCount(1)
        self.results_table.setItem(0, 0, QTableWidgetItem(""))
        self.results_table.setItem(0, 1, QTableWidgetItem(""))
        self.results_table.setItem(0, 2, QTableWidgetItem("暂无检测结果"))
        self.results_table.setItem(0, 3, QTableWidgetItem(""))
        self.results_table.setItem(0, 4, QTableWidgetItem(""))
        
        layout.addWidget(self.results_table)
        
        # 导出按钮
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        
        export_btn = QPushButton("导出结果")
        export_btn.clicked.connect(self.export_results)
        export_layout.addWidget(export_btn)
        
        layout.addLayout(export_layout)
        
        return panel
    
    def setup_connections(self):
        """设置信号连接"""
        # 滑块变化时自动重新检测（如果有图片的话）
        self.conf_slider.valueChanged.connect(self.auto_redetect)
        self.iou_slider.valueChanged.connect(self.auto_redetect)
    
    def load_model(self):
        """加载YOLO模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择YOLO模型文件",
            "",
            "模型文件 (*.pt *.onnx);;所有文件 (*)"
        )
        
        if file_path:
            if self.detector.load_model(file_path):
                self.status_label.setText(f"模型加载成功: {os.path.basename(file_path)}")
                self.select_image_btn.setEnabled(True)
                
                # 更新支持的鸟类列表
                classes_text = "支持的鸟类 (共{}种):\n".format(len(self.detector.class_names))
                lines = []
                for idx, class_en in enumerate(self.detector.class_names):
                    # 去掉编号前缀
                    if '.' in class_en:
                        class_en_key = class_en.split('.', 1)[1]
                    else:
                        class_en_key = class_en
                    class_cn = self.class_name_map.get(class_en_key, class_en_key)
                    lines.append(f"{idx+1}.{class_en_key}_{class_cn}")
                classes_text += "\n".join(lines)
                self.classes_text.setText(classes_text)
                
            else:
                QMessageBox.critical(self, "错误", "模型加载失败！")
    
    def select_image(self):
        """选择图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片文件",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        
        if file_path:
            if self.image_display.load_image(file_path):
                self.current_image_path = file_path
                self.detect_btn.setEnabled(True)
                self.status_label.setText("图片加载成功，可以开始检测")
                
                # 清除之前的检测结果
                # self.current_detections = []
                # self.update_results_table([])
                self.target_count_label.setText("0")
            else:
                QMessageBox.critical(self, "错误", "图片加载失败！")
    
    def start_detection(self):
        """开始检测"""
        if not self.current_image_path or not self.detector.model_loaded:
            return
        
        # 禁用按钮，显示进度
        self.detect_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 无限进度条
        
        # 获取当前参数
        conf_threshold = self.conf_slider.value() / 100.0
        iou_threshold = self.iou_slider.value() / 100.0
        
        # 创建检测线程
        self.detection_thread = DetectionThread(
            self.detector,
            self.current_image_path,
            conf_threshold,
            iou_threshold
        )
        
        # 连接信号
        self.detection_thread.detection_finished.connect(self.on_detection_finished)
        self.detection_thread.detection_progress.connect(self.on_detection_progress)
        
        # 开始检测
        self.detection_thread.start()
    
    def on_detection_finished(self, detections):
        """检测完成回调"""
        # 累加历史检测结果
        for det in detections:
            # 记录当前图片路径，方便导出
            det['image_path'] = self.current_image_path
            self.current_detections.append(det)
    
        # 在图片上绘制检测结果（传入中文映射表）
        self.image_display.draw_detections(detections, self.class_name_map)
    
        # 生成热力图
        if self.image_display.original_pixmap:
            self.heatmap_display.generate_heatmap(
                detections,
                self.image_display.original_pixmap.width(),
                self.image_display.original_pixmap.height()
            )
    
        # 更新结果表格，传入所有历史检测
        self.update_results_table(self.current_detections)
    
        # 更新目标数量（本次检测数）
        self.target_count_label.setText(str(len(detections)))
    
        # 恢复按钮状态
        self.detect_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
        # 更新状态
        self.status_label.setText(f"检测完成，发现 {len(detections)} 只鸟类")
    
    def on_detection_progress(self, message):
        """检测进度回调"""
        self.status_label.setText(message)
    
    def on_conf_changed(self, value):
        """置信度滑块变化"""
        conf_value = value / 100.0
        self.conf_label.setText(f"{conf_value:.2f}")
    
    def on_iou_changed(self, value):
        """IOU滑块变化"""
        iou_value = value / 100.0
        self.iou_label.setText(f"{iou_value:.2f}")
    
    def auto_redetect(self):
        """参数变化时自动重新检测"""
        if self.current_image_path and self.detector.model_loaded and not hasattr(self, 'detection_thread'):
            # 添加延时避免频繁检测
            if hasattr(self, 'redetect_timer'):
                self.redetect_timer.stop()
            
            self.redetect_timer = QTimer()
            self.redetect_timer.timeout.connect(self.start_detection)
            self.redetect_timer.setSingleShot(True)
            self.redetect_timer.start(500)  # 500ms延时
    
    def update_results_table(self, detections):
        """更新结果表格"""
        if not detections:
            self.results_table.setRowCount(1)
            self.results_table.setItem(0, 0, QTableWidgetItem(""))
            self.results_table.setItem(0, 1, QTableWidgetItem(""))
            self.results_table.setItem(0, 2, QTableWidgetItem("暂无检测结果"))
            self.results_table.setItem(0, 3, QTableWidgetItem(""))
            self.results_table.setItem(0, 4, QTableWidgetItem(""))
            return
    
        self.results_table.setRowCount(len(detections))
    
        for i, detection in enumerate(detections):
            # 序号
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
    
            # 图像信息
            img_name = os.path.basename(detection.get('image_path', '')) if detection.get('image_path') else "用户上传图片"
            self.results_table.setItem(i, 1, QTableWidgetItem(img_name))
    
            # 结果（英文转中文）
            class_en = detection['class']
            if '.' in class_en:
                class_en_key = class_en.split('.', 1)[1]
            else:
                class_en_key = class_en
            class_cn = self.class_name_map.get(class_en_key, class_en)
            self.results_table.setItem(i, 2, QTableWidgetItem(class_cn))
    
            # 位置
            bbox = detection['bbox']
            position = f"{int(bbox[0])},{int(bbox[1])},{int(bbox[2]-bbox[0])},{int(bbox[3]-bbox[1])}"
            self.results_table.setItem(i, 3, QTableWidgetItem(position))
    
            # 置信度
            confidence = f"{detection['confidence']:.1%}"
            self.results_table.setItem(i, 4, QTableWidgetItem(confidence))
    
        self.results_table.resizeColumnsToContents()
    
    def export_results(self):
        """导出检测结果"""
        if not self.current_detections:
            QMessageBox.information(self, "提示", "暂无检测结果可导出")
            return
    
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存检测结果",
            "bird_detection_results.csv",
            "CSV文件 (*.csv);;所有文件 (*)"
        )
    
        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["序号", "图像信息", "结果", "位置", "置信度"])
    
                    for i, detection in enumerate(self.current_detections):
                        img_name = os.path.basename(self.current_image_path) if self.current_image_path else "用户上传图片"
                        bbox = detection['bbox']
                        position = f"{int(bbox[0])},{int(bbox[1])},{int(bbox[2]-bbox[0])},{int(bbox[3]-bbox[1])}"
                        confidence = f"{detection['confidence']:.3f}"
    
                        # 导出中文名
                        class_en = detection['class']
                        if '.' in class_en:
                            class_en_key = class_en.split('.', 1)[1]
                        else:
                            class_en_key = class_en
                        class_cn = self.class_name_map.get(class_en_key, class_en)
    
                        writer.writerow([
                            i + 1,
                            img_name,
                            class_cn,
                            position,
                            confidence
                        ])
    
                QMessageBox.information(self, "成功", f"结果已导出到: {file_path}")
    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {e}")


def main():
    """主函数"""
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("鸟类识别系统")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("AI Detection Lab")
    
    # 设置样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = BirdDetectionApp()
    window.show()
    
    # 启动应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
