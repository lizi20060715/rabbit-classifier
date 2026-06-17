"""
Streamlit 在线部署 - 兔子品种图像分类器
"""

import streamlit as st
from fastai.vision.all import *
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="兔子品种图像分类器",
    page_icon="🐰",
    layout="centered"
)

@st.cache_resource
def load_model():
    """加载模型"""
    model_path = Path(__file__).parent / "models" / "resnet18_rabbit_classifier.pkl"
    
    try:
        learn = load_learner(model_path)
        return learn
    except Exception as e:
        # 如果 pickle 加载失败，尝试只加载权重
        st.warning("完整模型加载失败，尝试加载权重...")
        try:
            from torchvision.models import resnet18
            import torch
            
            # 创建模型
            model = resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 4)
            
            # 加载权重
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            
            # 创建 DataLoaders
            from fastai.vision.data import ImageDataLoaders
            from fastai.vision.augment import aug_transforms
            from fastai.vision.core import ImageBlock, CategoryBlock, Resize
            
            # 返回一个简单的包装对象
            class SimpleLearner:
                def __init__(self, model, vocab):
                    self.model = model
                    self.dls = type('obj', (object,), {'vocab': vocab})()
                
                def predict(self, img):
                    self.model.eval()
                    with torch.no_grad():
                        img_tensor = PILImage.create(img).resize((224,224)).to_tensor().unsqueeze(0)
                        output = self.model(img_tensor)
                        probs = torch.softmax(output, dim=1)[0]
                        pred_idx = probs.argmax().item()
                        return self.dls.vocab[pred_idx], pred_idx, probs
            
            vocab = ['波兰兔-Polish_Rabbit', '荷兰侏儒兔-Dutch_Dwarf_Rabbit', 
                    '荷兰垂耳兔-holland-lop', '迷你垂耳兔-mini-lop']
            return SimpleLearner(model, vocab)
            
        except Exception as e2:
            raise RuntimeError(f"无法加载模型: {e}\n{e2}")

# 主界面
st.title("🐰 兔子品种图像分类器")
st.markdown("上传一张兔子图片，AI将自动识别其品种！")

# 加载模型
with st.spinner("正在加载模型..."):
    try:
        learn = load_model()
        class_names = learn.dls.vocab
        st.success(f"✅ 模型加载成功！支持 {len(class_names)} 个品种")
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)[:200]}")
        st.info("💡 请检查模型文件是否正确上传。")
        st.stop()

# 显示类别
with st.expander("📋 支持的兔子品种"):
    for i, name in enumerate(class_names, 1):
        st.write(f"{i}. {name}")

# 文件上传
uploaded_file = st.file_uploader(
    "📤 上传兔子图片",
    type=['jpg', 'jpeg', 'png', 'bmp']
)

if uploaded_file:
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
                
                # 概率表格
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
