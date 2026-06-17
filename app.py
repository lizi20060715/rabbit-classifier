"""
Streamlit 在线部署 - 兔子品种图像分类器
运行方式: streamlit run app.py
"""

import streamlit as st
from fastai.vision.all import *
import numpy as np
from PIL import Image

# 页面配置
st.set_page_config(
    page_title="兔子品种图像分类器",
    page_icon="🐰",
    layout="centered"
)

# 加载模型
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "models" / "resnet18_rabbit_classifier.pkl"
    learn = load_learner(model_path)
    return learn

# 主界面
st.title("🐰 兔子品种图像分类器")
st.markdown("上传一张兔子图片，AI将自动识别其品种！")

# 加载模型（带加载提示）
with st.spinner("正在加载模型..."):
    try:
        learn = load_model()
        class_names = learn.dls.vocab
        st.success(f"✅ 模型加载成功！支持 {len(class_names)} 个品种")
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        st.stop()

# 显示支持的类别
with st.expander("📋 支持的兔子品种"):
    for i, name in enumerate(class_names, 1):
        st.write(f"{i}. {name}")

# 文件上传
uploaded_file = st.file_uploader(
    "📤 上传兔子图片",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    help="支持 JPG、PNG、BMP 格式"
)

if uploaded_file is not None:
    # 显示上传的图片
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图片", use_column_width=True)
    
    # 预测按钮
    if st.button("🔍 开始识别"):
        with st.spinner("正在分析..."):
            # 进行预测
            pred, pred_idx, probs = learn.predict(image)
            
            # 获取置信度
            confidence = probs[pred_idx].item()
            
            # 显示结果
            st.subheader("🎯 识别结果")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("预测品种", str(pred))
            with col2:
                st.metric("置信度", f"{confidence*100:.2f}%")
            
            # 显示所有类别的概率
            st.subheader("📊 各类别概率")
            
            # 排序概率
            prob_data = []
            for i, name in enumerate(class_names):
                prob_data.append({
                    "品种": name,
                    "概率": float(probs[i]),
                    "百分比": f"{float(probs[i])*100:.2f}%"
                })
            
            # 按概率排序
            prob_data.sort(key=lambda x: x["概率"], reverse=True)
            
            # 显示为表格
            st.dataframe(
                prob_data,
                column_config={
                    "品种": st.column_config.TextColumn("品种"),
                    "概率": st.column_config.ProgressColumn(
                        "概率",
                        min_value=0,
                        max_value=1,
                        format="%.4f"
                    ),
                    "百分比": st.column_config.TextColumn("百分比")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # 结论
            if confidence > 0.8:
                st.success(f"🎉 模型非常有信心这是 **{pred}**！")
            elif confidence > 0.5:
                st.info(f"💡 模型认为这可能是 **{pred}**，但置信度一般。")
            else:
                st.warning(f"⚠️ 模型不太确定，请尝试上传更清晰的图片。")

# 底部信息
st.markdown("---")
st.markdown("📚 **实验五 - CNN图像分类 | Fast.ai + ResNet18**")
st.markdown("🔧 本应用由 Streamlit 驱动部署")
