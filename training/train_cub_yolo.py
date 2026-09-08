#!/usr/bin/env python3
"""
基于CUB-200-2011数据集的YOLO训练脚本
使用您的数据集文件结构进行鸟类检测模型训练
"""

import os
import shutil
import pandas as pd
import yaml
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import torch

class CUBDatasetConverter:
    """CUB-200-2011数据集转换器"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.output_path = self.dataset_path / "yolo_format"
        
    def load_dataset_files(self):
        """加载数据集的各个标注文件"""
        print("正在加载数据集文件...")
        
        # 读取images.txt - 图片路径
        self.images_df = pd.read_csv(
            self.dataset_path / 'images.txt', 
            sep=' ', names=['img_id', 'filepath'], 
            header=None
        )
        
        # 读取bounding_boxes.txt - 边界框标注
        self.bbox_df = pd.read_csv(
            self.dataset_path / 'bounding_boxes.txt',
            sep=' ', names=['img_id', 'x', 'y', 'width', 'height'],
            header=None
        )
        
        # 读取image_class_labels.txt - 类别标签
        self.labels_df = pd.read_csv(
            self.dataset_path / 'image_class_labels.txt',
            sep=' ', names=['img_id', 'class_id'],
            header=None
        )
        
        # 读取train_test_split.txt - 训练/测试划分
        self.split_df = pd.read_csv(
            self.dataset_path / 'train_test_split.txt',
            sep=' ', names=['img_id', 'is_train'],
            header=None
        )
        
        # 读取classes.txt - 类别名称
        with open(self.dataset_path / 'classes.txt', 'r') as f:
            self.class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
        
        print(f"加载完成: {len(self.images_df)} 张图片, {len(self.class_names)} 个类别")
        
    def merge_data(self):
        """合并所有标注信息"""
        print("正在合并数据...")
        
        # 合并所有数据
        self.data = self.images_df.merge(self.bbox_df, on='img_id')
        self.data = self.data.merge(self.labels_df, on='img_id')
        self.data = self.data.merge(self.split_df, on='img_id')
        
        print(f"合并完成: {len(self.data)} 条记录")
        
    def create_yolo_structure(self):
        """创建YOLO格式的目录结构"""
        print("创建YOLO目录结构...")
        
        # 创建目录
        (self.output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
    def convert_annotations(self):
        """转换标注格式并复制文件"""
        print("正在转换标注格式...")
        
        train_count = 0
        val_count = 0
        
        for _, row in self.data.iterrows():
            # 获取原始图片路径
            src_img_path = self.dataset_path / "images" / row['filepath']
            
            # 检查图片是否存在
            if not src_img_path.exists():
                print(f"警告: 图片不存在 {src_img_path}")
                continue
                
            # 获取图片尺寸
            try:
                with Image.open(src_img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"无法读取图片 {src_img_path}: {e}")
                continue
            
            # 转换边界框格式 (绝对坐标 -> 相对坐标)
            x_center = (row['x'] + row['width'] / 2) / img_width
            y_center = (row['y'] + row['height'] / 2) / img_height
            width = row['width'] / img_width
            height = row['height'] / img_height
            
            # 确保边界框在有效范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # YOLO格式：class_id x_center y_center width height
            class_id = row['class_id'] - 1  # CUB从1开始，YOLO从0开始
            
            # 确定是训练集还是验证集
            split = "train" if row['is_train'] == 1 else "val"
            
            # 创建新的文件名（避免目录冲突）
            img_filename = f"{row['img_id']:06d}.jpg"
            txt_filename = f"{row['img_id']:06d}.txt"
            
            # 目标路径
            dst_img_path = self.output_path / "images" / split / img_filename
            dst_txt_path = self.output_path / "labels" / split / txt_filename
            
            # 复制图片
            shutil.copy2(src_img_path, dst_img_path)
            
            # 创建标注文件
            with open(dst_txt_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            if split == "train":
                train_count += 1
            else:
                val_count += 1
                
            if (train_count + val_count) % 1000 == 0:
                print(f"已处理: {train_count + val_count} 张图片")
        
        print(f"转换完成: 训练集 {train_count} 张, 验证集 {val_count} 张")
        
    def create_yaml_config(self):
        """创建YOLO配置文件"""
        print("创建YOLO配置文件...")
        
        config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': {i: name for i, name in enumerate(self.class_names)}
        }
        
        config_path = self.output_path / "cub200.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"配置文件已保存: {config_path}")
        return config_path
        
    def convert(self):
        """执行完整转换流程"""
        print("开始转换CUB-200-2011数据集...")
        
        self.load_dataset_files()
        self.merge_data()
        self.create_yolo_structure()
        self.convert_annotations()
        config_path = self.create_yaml_config()
        
        print("数据集转换完成!")
        return config_path


class YOLOTrainer:
    """YOLO训练器"""
    
    def __init__(self, config_path, model_size='s'):
        self.config_path = config_path
        self.model_size = model_size
        self.model_path = f'yolov8{model_size}.pt'
        
    def setup_training_environment(self):
        """设置训练环境"""
        print("设置训练环境...")
        
        # 检查GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        if self.device == 'cuda':
            print(f"GPU型号: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
    def train(self, epochs=100, batch_size=16, img_size=640):
        """开始训练"""
        print(f"开始训练YOLO{self.model_size.upper()}模型...")

        # 加载模型
        model = YOLO(self.model_path)

        # 只保留基础参数
        train_args = {
            'data': str(self.config_path),
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'device': self.device,
            'project': 'runs/detect',
            'name': f'cub200_yolo{self.model_size}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'SGD',
            'lr0': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'workers': 8,
            # 只保留这些，其他参数全部注释掉
        }

        print("训练参数:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")

        try:
            results = model.train(**train_args)
        except Exception as e:
            print(f"训练过程中发生异常: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

        print("训练完成，正在验证模型...")
        metrics = model.val()

        print(f"验证结果:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")

        return model, results, metrics
    
    def export_model(self, model, export_formats=['onnx', 'torchscript']):
        """导出训练好的模型"""
        print("导出模型...")
        
        for format in export_formats:
            try:
                if format == 'onnx':
                    model.export(format='onnx', dynamic=True, simplify=True)
                    print(f"✓ ONNX模型导出成功")
                elif format == 'torchscript':
                    model.export(format='torchscript')
                    print(f"✓ TorchScript模型导出成功")
                elif format == 'engine':
                    model.export(format='engine', device=0)
                    print(f"✓ TensorRT模型导出成功")
                elif format == 'coreml':
                    model.export(format='coreml')
                    print(f"✓ CoreML模型导出成功")
            except Exception as e:
                print(f"✗ {format}格式导出失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("CUB-200-2011 鸟类检测YOLO训练流程")
    print("=" * 60)
    
    # 配置路径
    dataset_path = os.environ.get("CUB_DATASET_PATH", "data/CUB_200_2011")

    # 检查数据集路径
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        print("请修改 dataset_path 变量为您的实际数据集路径")
        return
    
    try:
        # 步骤1: 数据集转换
        print("\n步骤1: 转换数据集格式")
        converter = CUBDatasetConverter(dataset_path)
        config_path = converter.convert()
        
        # 步骤2: 模型训练
        print("\n步骤2: 开始模型训练")
        trainer = YOLOTrainer(config_path, model_size='s')  # 可选择 n, s, m, l, x
        trainer.setup_training_environment()
        
        # 开始训练（可根据需要调整参数）
        model, results, metrics = trainer.train(
            epochs=100,      # 训练轮数
            batch_size=16,   # 批次大小（根据GPU内存调整）
            img_size=640     # 输入图片尺寸
        )
        
        # 步骤3: 模型导出
        print("\n步骤3: 导出模型")
        trainer.export_model(model, ['onnx', 'torchscript'])
        
        print("\n" + "=" * 60)
        print("训练流程完成!")
        print("=" * 60)
        
        # 打印最终结果
        print(f"\n最终模型性能:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"训练结果保存在: runs/detect/cub200_yolo{trainer.model_size}/")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


class ModelTester:
    """模型测试器"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def test_single_image(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """测试单张图片"""
        print(f"正在测试图片: {image_path}")
        
        # 推理
        results = self.model(image_path, conf=conf_threshold, iou=iou_threshold)
        
        # 处理结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                
                for i, (box, confidence, class_id) in enumerate(zip(xyxy, conf, cls)):
                    class_name = self.model.names[int(class_id)]
                    detections.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': box.tolist()
                    })
                    
                    print(f"检测到: {class_name}, 置信度: {confidence:.3f}")
                    print(f"位置: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        
        if not detections:
            print("未检测到任何鸟类")
        
        return detections
    
    def batch_test(self, test_dir, output_dir="test_results"):
        """批量测试图片"""
        import glob
        
        os.makedirs(output_dir, exist_ok=True)
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        test_images = []
        
        for ext in image_extensions:
            test_images.extend(glob.glob(os.path.join(test_dir, ext)))
            test_images.extend(glob.glob(os.path.join(test_dir, ext.upper())))
        
        print(f"找到 {len(test_images)} 张测试图片")
        
        all_results = []
        for img_path in test_images:
            detections = self.test_single_image(img_path)
            all_results.append({
                'image': os.path.basename(img_path),
                'detections': detections
            })
            
            # 保存结果图片
            results = self.model(img_path)
            for i, result in enumerate(results):
                result.save(filename=os.path.join(output_dir, f"result_{os.path.basename(img_path)}"))
        
        return all_results


def quick_test():
    """快速测试函数"""
    print("快速测试模式")
    
    # 使用训练好的模型
    model_path = os.environ.get("BIRD_MODEL_PATH", "models/cub200-yolov8s-best.pt")

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先完成训练或修改模型路径")
        return
    
    # 创建测试器
    tester = ModelTester(model_path)
    
    # 测试单张图片（修改为实际图片路径）
    test_image = os.environ.get("BIRD_TEST_IMAGE", "data/test-image.jpg")
    if os.path.exists(test_image):
        detections = tester.test_single_image(test_image)
        print(f"检测结果: {detections}")
    else:
        print(f"测试图片不存在: {test_image}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CUB-200-2011 YOLO训练脚本")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                      help="运行模式: train(训练) 或 test(测试)")
    parser.add_argument("--dataset", type=str, default="data/CUB_200_2011",
                      help="数据集路径")
    parser.add_argument("--model_size", choices=["n", "s", "m", "l", "x"], default="s",
                      help="模型尺寸")
    parser.add_argument("--epochs", type=int, default=100,
                      help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="批次大小")
    parser.add_argument("--img_size", type=int, default=640,
                      help="输入图片尺寸")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # 设置训练参数
        dataset_path = args.dataset
        
        # 执行训练
        if os.path.exists(dataset_path):
            print(f"使用数据集: {dataset_path}")
            print(f"模型尺寸: YOLO{args.model_size.upper()}")
            print(f"训练轮数: {args.epochs}")
            print(f"批次大小: {args.batch_size}")
            
            # 更新主函数中的参数
            converter = CUBDatasetConverter(dataset_path)
            config_path = converter.convert()
            
            trainer = YOLOTrainer(config_path, model_size=args.model_size)
            trainer.setup_training_environment()
            
            model, results, metrics = trainer.train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=args.img_size
            )
            
            trainer.export_model(model)
        else:
            print(f"数据集路径不存在: {dataset_path}")
    
    elif args.mode == "test":
        quick_test()
    
    else:
        main()
