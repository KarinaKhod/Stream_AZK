import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
import time

st.set_page_config(page_title="MobileNetV3: готовая модель", layout="wide")
# Названия классов
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

@st.cache_resource
def load_model():
    # Загружаем всю модель целиком (torch.save(model))
    model = torch.load("mobilenet-trained-6cl.pt", map_location="cpu", weights_only=False)
    model.eval()
    return model

model = load_model()

# Преобразования изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


st.title("Классификация изображений с использованием сохранённой модели")

source = st.radio("Источник изображения:", ["Файл", "Ссылка"])
images = []

if source == "Файл":
    uploaded = st.sidebar.file_uploader("Загрузите изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            img = Image.open(f).convert("RGB")
            images.append(img)

elif source == "Ссылка":
    url = st.text_input("Вставьте ссылку на изображение")
    if url:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(img)
        except:
            st.error("Ошибка загрузки изображения.")

if images:
    for img in images:
        st.image(img, caption="Загруженное изображение", use_column_width=True)
        input_tensor = transform(img).unsqueeze(0)

        start = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        elapsed = time.time() - start

        _, pred = torch.max(output, 1)
        st.success(f"Класс: {class_names[pred.item()]}")
        st.info(f"Время обработки: {elapsed:.3f} сек")