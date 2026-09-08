# 来源档案清单

此资料夹由下列本地资料整理而来，原始档案均未移动或删除。

## 已复制并整理

- `final/bird_detection_app.py` → `src/bird_detection_app.py`
- `final/final.py` → `training/train_cub_yolo.py`
- `final/CUB_200_2011/classes.txt` → `data/classes.txt`
- `final/runs/detect/cub200_yolos/weights/best.pt` → `models/cub200-yolov8s-best.pt`
- `final/runs/detect/cub200_yolos/` 的统计图与 `results.csv` → `results/`
- 期中、期末 PDF 与课程简报 → `docs/course-reports/`

## 刻意未复制

- `final/bird_detection_env/`：可由 `requirements.txt` 重建。
- `final/CUB_200_2011/images/` 与转换后的图片：体积大，且原始照片权利应另行确认。
- `final/runs/**/train_batch*.jpg` 与 `val_batch*.jpg`：含资料集原始照片。
- `1.py`、`bird_detection_app copy.py`、`hotfix.py`：重复或中间版本。
- `yolov8s.pt`、`yolo11n.pt`：可由上游取得的预训练权重。
- `best.onnx`、`best.torchscript`：与最佳 `.pt` 权重功能重复且体积较大。

