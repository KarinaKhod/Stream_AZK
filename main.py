import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
import time

st.set_page_config(page_title="Классификатор", layout="wide")
st.write("Выберите страницу слева: MobileNet или EfficientNet.")