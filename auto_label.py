from roboflow import Roboflow
from ultralytics import YOLO
import os

# 1. 載入模型
print("載入 YOLO 模型中...")
model = YOLO("./runs/segment/0424_yolo_runs_v1/v1_seed_model/weights/best.pt")

# 2. 自動生成 classes.txt
classes_path = "./classes.txt"
with open(classes_path, "w", encoding="utf-8") as f:
    for i in range(len(model.names)):
        f.write(f"{model.names[i]}\n")
print(f"已生成類別對照表: {classes_path}")

# 3. 初始化 Roboflow
rf = Roboflow(api_key="ZaWZscFTeQdbGPVovpQT") 
project = rf.workspace("0621-dsqyp").project("2026_0409_asrlab215")

image_folder = "/home/asrlab_3090/yolo_bowei/dataset/rgb/"

print("開始本地推論與上傳...")
for filename in os.listdir(image_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        
        # 🔥 加入了你原本 CLI 的神準參數：imgsz=640 與 retina_masks=True
        results = model.predict(
            image_path, 
            conf=0.35, 
            imgsz=640,
            retina_masks=True,
            save_txt=True, 
            project="yolo11_predict", # 跟你原本的指令統一
            name="auto_labels", 
            exist_ok=True,
            verbose=False
        )
        
        # 🔥 關鍵修正：讓程式自己去問 YOLO 檔案到底存去哪裡了！
        save_dir = results[0].save_dir
        txt_filename = filename.rsplit(".", 1)[0] + ".txt"
        txt_path = os.path.join(save_dir, "labels", txt_filename)
        
        # 4. 防呆上傳機制
        if os.path.exists(txt_path):
            project.single_upload(
                image_path=image_path,
                annotation_path=txt_path,
                annotation_labelmap=classes_path
            )
            print(f"✅ 成功上傳並標註: {filename}")
        else:
            project.single_upload(image_path=image_path)
            print(f"⚠️ 上傳空圖片 (無預測物件): {filename}")

print("🎉 全部自動標註並上傳完畢！趕快去 Roboflow 網頁檢查吧！")
