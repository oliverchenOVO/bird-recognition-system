# CUB-200 鳥類偵測與辨識系統

這是一個以 YOLOv8s 與 CUB-200-2011 資料集完成的桌面鳥類辨識系統。系統可以載入圖片、偵測鳥類位置、辨識 200 個鳥種、顯示信賴度與熱力圖，並將結果匯出為 CSV。

## 成果摘要

保存的 100 epochs 訓練紀錄顯示：

| 指標 | 結果 |
|---|---:|
| Precision | 0.8097 |
| Recall | 0.7631 |
| mAP@50 | **0.8265** |
| mAP@50–95 | **0.7269** |

![訓練指標](results/results.png)

## 主要功能

- PyQt6 桌面圖形介面
- YOLO `.pt` 與 `.onnx` 模型載入
- 單張圖片鳥類偵測與 200 類鳥種辨識
- CONF 與 IOU 閾值調整
- 偵測框、中文鳥名、信賴度與熱力圖顯示
- 偵測結果 CSV 匯出
- CUB-200-2011 標註轉換、訓練與模型匯出流程

## 資料夾結構

```text
.
├── src/                 # 桌面辨識應用程式
├── training/            # CUB-200 轉 YOLO 與訓練程式
├── models/              # 保存的最佳訓練權重
├── data/classes.txt     # 200 個類別名稱（不含圖片資料集）
├── results/             # 訓練指標、混淆矩陣與輸出範例
└── docs/course-reports/ # 期中、期末報告與簡報
```

## 執行桌面應用程式

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\bird_detection_app.py
```

啟動後按「載入模型」，選擇：

```text
models/cub200-yolov8s-best.pt
```

再選擇一張圖片並開始偵測。

## 重新訓練

本倉庫不包含 CUB-200-2011 圖片資料集。取得資料集後，可執行：

```powershell
python training\train_cub_yolo.py --mode train --dataset "D:\path\to\CUB_200_2011" --model_size s --epochs 100 --batch_size 16 --img_size 640
```

訓練流程會讀取原始 CUB 標註，將邊界框轉換成 YOLO 格式，再訓練模型並輸出評估結果。

## 訓練結果

- [完整訓練紀錄（CSV）](results/results.csv)
- [混淆矩陣](results/confusion_matrix.png)
- [正規化混淆矩陣](results/confusion_matrix_normalized.png)
- [Precision–Recall 曲線](results/PR_curve.png)
- [偵測結果輸出範例](results/sample-detection.csv)
- [期末報告](docs/course-reports/final-report.pdf)
- [期中報告](docs/course-reports/midterm-report.pdf)
- [課程簡報](docs/course-reports/presentation.pptx)

## 資料與授權說明

CUB-200-2011 原始圖片、Python 虛擬環境、訓練快取和重複程式版本沒有放進這個資料夾。公開倉庫前，請先閱讀 [公開檢查清單](docs/PUBLICATION_CHECKLIST.md)。目前未加入開源授權條款；若公開發布，在選擇授權條款前仍應確認程式、訓練權重、課程報告圖片與第三方套件的使用條件。
