# 🐶🐱 Dogs vs Cats Classification with Grad-CAM

使用 VGG16、ResNet50、EfficientNetB0 三種預訓練模型進行貓狗分類，並以 Grad-CAM 技術可視化模型判斷依據。

## 📌 內容包含

- 📂 資料整理（來自 Kaggle）
- 📊 模型比較（準確率與訓練過程）
- 🔥 Grad-CAM 可視化（聚焦區域分析）
- ✅ 預測結果與 submission 輸出

## 🧠 模型架構

- 載入 ImageNet 預訓練模型
- 接上 GAP → Dense(256) → Dropout → Sigmoid

## 🧪 模型效能比較

| 模型         | Val Accuracy | Grad-CAM 聚焦品質 |
|--------------|--------------|-------------------|
| VGG16        | ⭐ 約 92%     | 中等清晰          |
| ResNet50     | 約 63%       | 最佳聚焦（貓臉）   |
| EfficientNet | 約 50%       | 幾乎無熱區        |

## 🔍 可視化結果

![](images/vgg16_gradcam.jpg)
![](images/resnet50_gradcam.jpg)

## 📈 訓練曲線

![](images/acc_loss_curve.png)

## 📁 Colab Notebook

👉 [查看原始程式碼（ipynb）](./貓狗圖像分類與Grad_CAM可視化.ipynb)

## 🙌 作者

Ricky Chen  
Email: rickychen0606@gmail.com

