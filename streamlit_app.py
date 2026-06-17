"""
Streamlit 在线部署 - 兔子品种图像分类器
"""

import streamlit as st
from fastai.vision.all import *
from pathlib import Path

st.set_page_config(
    page_title="兔子品种图像分类器",
    page_icon="🐰",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "models" / "resnet18_rabbit_classifier.pkl"
    if not model_path.exists():
        st.error(f"模型文件不存在：{model_path}")
        st.stop()
    try:
        learn = load_learner(model_path)
        return learn
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()

# 后续代码保持不变...
# （主界面、上传、预测等不变）