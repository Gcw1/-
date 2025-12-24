# 确保导入路径完整且兼容
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_traffic_model(input_shape=(32, 32, 3), num_classes=43):
    """
    构建交通标志识别CNN模型
    :param input_shape: 输入图像形状 (高, 宽, 通道)
    :param num_classes: 分类数（GTSRB为43）
    :return: 编译好的模型
    """
    # 明确使用keras.Sequential，避免未解析引用
    model = keras.Sequential([
        # 卷积层1 + 池化层
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),  # 新增：加速训练
        layers.MaxPooling2D((2, 2)),

        # 卷积层2 + 池化层
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 卷积层3 + 池化层
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 全连接层
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # 防止过拟合
        layers.Dense(num_classes, activation='softmax')
    ])

    # 编译模型（明确使用keras的优化器/损失函数）
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')]
    )
    return model


# 测试模型构建（验证无报错）
if __name__ == '__main__':
    model = build_traffic_model()
    model.summary()  # 打印模型结构
    print("模型构建成功，无未解析引用错误！")