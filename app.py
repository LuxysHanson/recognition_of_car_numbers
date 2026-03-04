import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
try:
    import langchain
except ImportError:
    pass
else:
    # Создаем фиктивный модуль, если его нет
    if not hasattr(langchain, 'docstore'):
        from langchain_community import docstore
        sys.modules['langchain.docstore'] = docstore

import streamlit as st
import cv2
from paddleocr import PaddleOCR
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import tempfile
from PIL import Image
import re

# Настройка страницы
st.set_page_config(page_title="ANPR System", page_icon="🚗", layout="wide")

# Загрузка моделей с кэшированием
@st.cache_resource
def load_models():
    # Замените 'best.pt' на путь к вашей модели
    with st.spinner("Загрузка YOLO..."):
        model = YOLO('models/best_medium.pt')
    with st.spinner("Загрузка PaddleOCR..."):
        reader = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
    return model, reader

model, reader = load_models()

# Sidebar - Настройки
st.sidebar.title("⚙️ Настройки")
conf_threshold = st.sidebar.slider("Порог уверенности YOLO", 0.1, 1.0, 0.5, 0.05)
save_csv = st.sidebar.checkbox("Сохранять историю в CSV", value=True)

# Главный интерфейс
st.title("🚗 Система распознавания номеров")
st.write("Мини-проект для автоматической детекции и OCR автомобильных номеров.")

tab1, tab2 = st.tabs(["🖼 Загрузка фото", "🎥 Загрузка видео"])

# --- ЛОГИКА ОБРАБОТКИ ---
def process_frame(frame, conf):
    results = model(frame, conf=conf, iou=0.45, verbose=False)[0]
    data = []
    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = result
        plate = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if plate.size > 0:
            h, w = plate.shape[:2]
            # Если высота больше ширины — это вертикальный номер
            if h > w:
                plate = cv2.rotate(plate, cv2.ROTATE_90_CLOCKWISE)
            
            plate_scaled = cv2.resize(plate, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            ocr_res = reader.ocr(plate_scaled)

            # Проверяем структуру PaddleX
            if isinstance(ocr_res, list) and len(ocr_res) > 0:
 
                res_dict = ocr_res[0] # Берем первый элемент списка
                
                # Извлекаем список текстов и их скоров
                texts = res_dict.get('rec_texts', [])
                scores = res_dict.get('rec_scores', [])

                raw_combined = "".join(texts) 
                plate_text = format_kz_plate(raw_combined)

                for text, prob in zip(texts, scores):
                    # Игнорируем надпись 'KZ', она нам не нужна
                    if text.upper() == 'KZ':
                        continue
                        
                    print(f"Найдено в PaddleX: {text} (score: {prob})")

                    # Если в номере есть цифры и он прошел фильтр
                    if any(char.isdigit() for char in plate_text) and prob >= conf:
                        data.append({
                            "Время": datetime.now().strftime("%H:%M:%S"), 
                            "Номер": plate_text, 
                            "Уверенность": round(float(prob), 2)
                        })
                        
                        # Отрисовка на кадре
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                        cv2.putText(frame, plate_text, (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame, data


def format_kz_plate(raw_text):

    if not raw_text or len(raw_text) < 4:
        return ""
    
    # 1. Оставляем только буквы и цифры, убираем мусор вроде ] [ /
    clean = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

    # Шаблон для вертикальных (3 цифры + 2 цифры региона + 3 буквы)
    # Пример: 03901AJM -> 039 AJM 01
    match_vert = re.search(r'(\d{3})(\d{2})([A-Z]{3})', clean)
    if match_vert:
        num, reg, lets = match_vert.groups()
        return f"{num} {lets} {reg}"
    
    # --- ШАБЛОН 1: Госномера (только цифры: 005 000 05) ---
    match_gov = re.search(r'(\d{3})(\d{3})(\d{2})', clean)
    if match_gov:
        num1, num2, reg = match_gov.groups()
        return f"{num1} {num2} {reg}"
    
    # --- ШАБЛОН 2: Частные (3 цифры + 3 буквы + 2 цифры: 555 HAA 09) ---
    match_priv = re.search(r'(\d{3})([A-Z]{3})(\d{2})', clean)
    if match_priv:
        num, lets, reg = match_priv.groups()
        # Если регион считался одной цифрой (9), добавим 0 (09)
        reg = reg.zfill(2) 
        return f"{num} {lets} {reg}"
    
    # --- ШАБЛОН 3: Юрлица (3 цифры + 2 буквы + 2 цифры: 696 EJ 02) ---
    match_jur = re.search(r'(\d{3})([A-Z]{2})(\d{1,2})', clean)
    if match_jur:
        num, lets, reg = match_jur.groups()
        reg = reg.zfill(2)
        return f"{num} {lets} {reg}"

    return clean

# --- ТАБ 1: ФОТО ---
with tab1:
    img_file = st.file_uploader("Выберите изображение...", type=['jpg', 'jpeg', 'png', 'webp'], key="img")
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        processed_img, res_list = process_frame(image.copy(), conf_threshold)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(processed_img, channels="BGR", caption="Результат обработки")
        with col2:
            st.subheader("📝 Результаты")
            if res_list and save_csv:
                st.table(pd.DataFrame(res_list))
            else:
                st.info("Номера не найдены")

# --- ТАБ 2: ВИДЕО ---
with tab2:
    vid_file = st.file_uploader("Загрузите видеофайл...", type=['mp4', 'avi', 'mov'], key="vid")
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        run_btn = st.button("▶ Запустить обработку видео")

        # Создаем 3 колонки: пустая, для видео (центр), пустая
        col_side1, col_video, col_side2 = st.columns([1, 2, 1]) 

        with col_video:
            st_frame = st.empty() # Резервируем место под видео в центральной колонке

        st_table = st.empty() 
        
        all_video_data = []

        if run_btn:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Обработка каждого 5 кадра для производительности
                if frame_count % 5 == 0:
                    processed_frame, frame_data = process_frame(frame, conf_threshold)
                    all_video_data.extend(frame_data)
                    
                    st_frame.image(processed_frame, channels="BGR", use_container_width=True)
                    
                    if all_video_data:
                        df = pd.DataFrame(all_video_data).drop_duplicates(subset=['Номер']).tail(5)
                        st_table.dataframe(df, use_container_width=True)

                frame_count += 1
            
            cap.release()
            st.success("Обработка видео завершена!")
            
            if save_csv and all_video_data:
                final_df = pd.DataFrame(all_video_data).drop_duplicates(subset=['Номер'])
                st.download_button("📥 Скачать CSV отчет", final_df.to_csv(index=False), "report.csv", "text/csv")
