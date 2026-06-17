"""
Streamlit 在线部署 - 兔子品种图像分类器
"""

import streamlit as st
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np

# 页面配置
st.set_page_config(
    page_title="兔子品种图像分类器",
    page_icon="🐰",
    layout="centered"
)

@st.cache_resource
def load_model():
    """
    加载 PyTorch 模型权重（.pth 文件）
    优先查找 models/resnet18_weights.pth，再尝试根目录的 resnet18权重.pth
    """
    # 定义可能的路径（按优先级）
    possible_paths = [
        Path(__file__).parent / "models" / "resnet18_weights.pth",   # 推荐（英文命名）
        Path(__file__).parent / "resnet18权重.pth",                  # 您当前的文件
        Path(__file__).parent / "models" / "resnet18权重.pth",       # 也尝试 models 下的中文名
    ]
    
    model_path = None
    for p in possible_paths:
        if p.exists():
            model_path = p
            break
    
    if model_path is None:
        raise FileNotFoundError(
            "未找到模型文件！请确保将 'resnet18权重.pth' 放在项目根目录，"
            "或重命名为 'resnet18_weights.pth' 并放入 'models/' 文件夹。"
        )
    
    try:
        # 1. 创建模型结构（与训练时一致）
        model = resnet18(weights=None)
        num_classes = 4
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        # 2. 加载权重
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        # 3. 类别名称（必须与训练顺序一致）
        vocab = [
            '波兰兔-Polish_Rabbit',
            '荷兰侏儒兔-Dutch_Dwarf_Rabbit',
            '荷兰垂耳兔-holland-lop',
            '迷你垂耳兔-mini-lop'
        ]
        
        # 4. 图像预处理（与训练时保持一致）
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 5. 包装预测类（兼容原接口）
        class SimpleLearner:
            def __init__(self, model, vocab, transform):
                self.model = model
                self.dls = type('obj', (object,), {'vocab': vocab})()
                self.transform = transform
            
            def predict(self, img):
                """
                输入：PIL Image 或文件路径
                返回：(预测标签, 类别索引, 概率张量)
                """
                if isinstance(img, (str, Path)):
                    img = Image.open(img).convert('RGB')
                elif isinstance(img, Image.Image):
                    img = img.convert('RGB')
                else:
                    raise TypeError("输入必须是 PIL.Image 或文件路径")
                
                # 预处理
                img_tensor = self.transform(img).unsqueeze(0)  # [1, 3, 224, 224]
                
                self.model.eval()
                with torch.no_grad():
                    output = self.model(img_tensor)
                    probs = torch.softmax(output, dim=1)[0]   # [num_classes]
                    pred_idx = probs.argmax().item()
                    return self.dls.vocab[pred_idx], pred_idx, probs
        
        return SimpleLearner(model, vocab, transform)
    
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")

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
        st.error(f"❌ 模型加载失败: {str(e)}")
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
    image = Image.open(uploaded_file).convert('RGB')
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
st.markdown("📚 **实验五 - CNN图像分类 | PyTorch + ResNet18**")