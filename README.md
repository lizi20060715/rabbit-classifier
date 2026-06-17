# 🐰 兔子品种图像分类器

基于 Fast.ai + ResNet18 实现的图像分类器，支持4个兔子品种识别。

## 🚀 快速部署

### Streamlit Cloud 部署

1. Fork 本仓库
2. 访问 [share.streamlit.io](https://share.streamlit.io)
3. 连接仓库，主文件选择 `app.py`
4. 点击 Deploy

### 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 模型性能

- 测试准确率：75%
- 支持类别：4个品种
- 模型大小：~47MB

## 🐰 支持的品种

1. 波兰兔 (Polish Rabbit)
2. 荷兰侏儒兔 (Dutch Dwarf Rabbit)
3. 荷兰垂耳兔 (Holland Lop)
4. 迷你垂耳兔 (Mini Lop)

## 📁 项目结构

```
├── app.py                          # Streamlit 部署代码
├── requirements.txt                # 依赖列表
├── .gitignore                     # Git 忽略配置
├── README.md                      # 项目说明
└── models/
    └── resnet18_rabbit_classifier.pkl  # 训练好的模型
```
