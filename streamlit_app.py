"""
Streamlit 在线部署 - 兔子品种图像分类器
"""
import sys
import traceback
try:
    import streamlit as st
except Exception as e:
    print("CRITICAL STARTUP ERROR:", e)
    traceback.print_exc()
    raise
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
        st.error(f"模型文件不存在，请确保 {model_path} 已上传。")
        st.stop()
    try:
        learn = load_learner(model_path)
        return learn
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()

st.title("🐰 兔子品种图像分类器")
st.markdown("上传一张兔子图片，AI将自动识别其品种！")

with st.spinner("正在加载模型..."):
    learn = load_model()
    class_names = learn.dls.vocab
    st.success(f"✅ 模型加载成功！支持 {len(class_names)} 个品种")

with st.expander("📋 支持的兔子品种"):
    for i, name in enumerate(class_names, 1):
        st.write(f"{i}. {name}")

uploaded_file = st.file_uploader(
    "📤 上传兔子图片",
    type=['jpg', 'jpeg', 'png', 'bmp']
)

if uploaded_file:
    from PIL import Image
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图片", use_column_width=True)
    if st.button("🔍 开始识别"):
        with st.spinner("正在分析..."):
            try:
                pred, pred_idx, probs = learn.predict(image)
                confidence = probs[pred_idx].item()
                st.subheader("🎯 识别结果")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("预测品种", str(pred))
                with col2:
                    st.metric("置信度", f"{confidence*100:.2f}%")
                prob_data = [
                    {"品种": name, "概率": float(probs[i]), "百分比": f"{float(probs[i])*100:.2f}%"}
                    for i, name in enumerate(class_names)
                ]
                prob_data.sort(key=lambda x: x["概率"], reverse=True)
                st.dataframe(
                    prob_data,
                    column_config={
                        "品种": st.column_config.TextColumn("品种"),
                        "概率": st.column_config.ProgressColumn("概率", min_value=0, max_value=1),
                        "百分比": st.column_config.TextColumn("百分比")
                    },
                    hide_index=True
                )
            except Exception as e:
                st.error(f"预测失败: {e}")

st.markdown("---")
st.markdown("📚 **实验五 - CNN图像分类 | Fast.ai + ResNet18**")