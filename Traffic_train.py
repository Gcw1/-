import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import pandas as pd
import cv2  # 新增：用于图像缩放

# ========== 修复核心：读取图像后统一尺寸 ==========
def read_ppm_manually(file_path):
    """手动解析PPM文件，返回RGB图像"""
    try:
        with open(file_path, 'rb') as f:
            header = f.readline().decode('ascii').strip()
            if header not in ('P3', 'P6'):
                return None
            while True:
                line = f.readline().decode('ascii').strip()
                if not line.startswith('#'):
                    break
            width, height = map(int, line.split())
            max_val = int(f.readline().decode('ascii').strip())
            if header == 'P6':
                data = f.read(width * height * 3)
                img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            else:
                data = []
                while len(data) < width * height * 3:
                    data += list(map(int, f.readline().decode('ascii').split()))
                img = np.array(data, dtype=np.uint8).reshape((height, width, 3))
        return img
    except Exception as e:
        print(f"读取{file_path}失败：{e}")
        return None

def readTrafficSigns(rootpath, target_size=(32, 32)):  # 新增target_size参数
    """读取图像并统一缩放为target_size"""
    images = []
    labels = []
    for c in range(43):
        prefix = os.path.join(rootpath, format(c, '05d'))
        gt_file = os.path.join(prefix, 'GT-' + format(c, '05d') + '.csv')
        if not os.path.exists(gt_file):
            print(f"跳过标签文件：{gt_file}")
            continue
        gt_df = pd.read_csv(gt_file, sep=';')
        for idx, row in gt_df.iterrows():
            img_path = os.path.join(prefix, row['Filename'])
            if not os.path.exists(img_path):
                print(f"跳过图像文件：{img_path}")
                continue
            img = read_ppm_manually(img_path)
            if img is not None:
                # 核心修复：统一缩放图像尺寸
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                images.append(img)
                labels.append(row['ClassId'])
    # 转换为numpy数组（此时所有图像尺寸一致）
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    print(f"读取完成：{images.shape} 张图像，{len(np.unique(labels))} 类标签")
    return images, labels

# ========== 主程序 ==========
# 数据集路径（替换为你的真实路径）
train_image_dir = r'D:\OneDrive\桌面\Traffic_Sign_Classify\GTSRB\Final_Training\Images'
# 模型保存路径
MODEL_SAVE_PATH = r'D:\OneDrive\桌面\Traffic_Sign_Classify\Mymodle'

# 读取数据（统一缩放为32×32）
images, ys = readTrafficSigns(train_image_dir, target_size=(32, 32))

# 图像归一化（0-1范围）
images = images / 255.0

# 标签独热编码
ys = keras.utils.to_categorical(ys, num_classes=43)

# 划分训练集/验证集
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    images, ys, test_size=0.2, random_state=42, shuffle=True
)

# 构建模型
def build_model(input_shape=(32, 32, 3), num_classes=43):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

# 初始化模型
model = build_model(input_shape=(32, 32, 3), num_classes=43)
model.summary()

# 训练模型
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,  # 先训练10轮验证，后续可增加
    validation_data=(x_val, y_val),
    verbose=1
)

# 保存模型
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model.save(os.path.join(MODEL_SAVE_PATH, 'traffic_sign_model.h5'))
print(f"模型已保存至：{MODEL_SAVE_PATH}")

# 绘制训练曲线
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换为系统中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.legend()
plt.title('准确率变化')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.legend()
plt.title('损失变化')
plt.tight_layout()
plt.show()