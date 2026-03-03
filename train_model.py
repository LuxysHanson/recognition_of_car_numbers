import os
import shutil
from roboflow import Roboflow
from ultralytics import YOLO

from dotenv import load_dotenv
load_dotenv()

# Загрузка датасета KZ Carplate с Roboflow
rf = Roboflow(api_key = os.getenv('ROBOFLOW_API_KEY')) # Ключ в настройках Roboflow
project = rf.workspace("s-workspace-wg9tg").project("kz-carplate-numbers-lmqof")
version = project.version(2)
dataset = version.download("yolov8")

# Обучение модели
model = YOLO('yolov8m.pt')
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=1280,
    batch=-1, # Автоматический подбор размера батча под вашу память
    plots=True
)

# Сохраняем модель
target_dir = 'models'
os.makedirs(target_dir, exist_ok=True) # Создаем папку, если её нет

best_model_path = os.path.join(model.trainer.save_dir, 'weights', 'best_medium.pt')

if os.path.exists(best_model_path):
    shutil.copy(best_model_path, f"{target_dir}/best_medium.pt")
    print(f"✅ Модель скопирована в {target_dir}/best_medium.pt")
else:
    print("❌ Файл best_medium.pt не найден в директории обучения")