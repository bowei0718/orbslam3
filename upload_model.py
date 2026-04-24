# upload_weights.py
from roboflow import Roboflow

# 1. 一樣放入你的專屬 API Key
rf = Roboflow(api_key="ZaWZscFTeQdbGPVovpQT")

# 2. 定位到你的專案
project = rf.workspace("0621-dsqyp").project("2026_0409_asrlab215")
version = project.version(2) # 指定要綁定到哪一個 Version

# 3. 執行上傳 (假設你的權重檔產出在 runs/detect/train/weights/)
# 注意：model_path 是填寫「包含 best.pt 的資料夾路徑」，不是檔案本身
# version.deploy(model_type="yolov8", model_path="runs/detect/train/weights/")
version.deploy(model_type="yolov11", model_path="./runs/segment/0424_yolo_runs_v1/v1_seed_model/")

print("權重檔上傳完成！現在可以去 Roboflow 網頁使用 Auto-Label 了。")
