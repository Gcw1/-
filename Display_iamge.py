import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os  # 新增：处理文件路径
import pandas as pd  # 新增：读取标签文件

# 解决matplotlib中文显示问题（可选，避免标签中文乱码）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ========== 新增：手动解析PPM文件函数（核心修复） ==========
def read_ppm_manually(file_path):
    """纯Python解析PPM文件（支持P3/P6格式），绕过cv2.imread的兼容问题"""
    try:
        with open(file_path, 'rb') as f:
            # 读取PPM头部标识（P3=文本格式，P6=二进制格式）
            header = f.readline().decode('ascii').strip()
            if header not in ('P3', 'P6'):
                return None  # 仅支持标准PPM格式

            # 跳过注释行（以#开头的行）
            while True:
                line = f.readline().decode('ascii').strip()
                if not line.startswith('#'):
                    break

            # 读取宽、高、最大像素值
            width, height = map(int, line.split())
            max_val = int(f.readline().decode('ascii').strip())

            # 读取像素数据并转换为numpy数组
            if header == 'P6':
                # P6格式：二进制流，每个像素3字节（R/G/B）
                data = f.read(width * height * 3)
                img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            else:
                # P3格式：文本格式，按空格分割像素值
                data = []
                while len(data) < width * height * 3:
                    data += list(map(int, f.readline().decode('ascii').split()))
                img = np.array(data, dtype=np.uint8).reshape((height, width, 3))
        return img
    except Exception as e:
        # 打印读取失败的文件路径（便于排查）
        print(f"警告：读取{file_path}失败 → {e}")
        return None


def readTrafficSigns(rootpath):
    """
    替代原read_iamge.py中的函数：读取GTSRB训练集图像和标签
    输入：GTSRB训练集根目录（如Final_Training/Images）
    输出：images（图像列表）、labels（标签列表）
    """
    images = []  # 存储图像
    labels = []  # 存储标签
    # 遍历每个分类文件夹（00000, 00001...）
    for c in range(43):  # GTSRB共43类交通标志
        prefix = os.path.join(rootpath, format(c, '05d'))  # 拼接分类文件夹路径（如00000）
        gt_file = os.path.join(prefix, 'GT-' + format(c, '05d') + '.csv')  # 标签文件路径

        # 新增：检查标签文件是否存在（避免报错）
        if not os.path.exists(gt_file):
            print(f"警告：标签文件不存在 → {gt_file}")
            continue

        # 读取标签文件（GTSRB的csv用分号分隔）
        gt_df = pd.read_csv(gt_file, sep=';')

        # 遍历该分类下的所有图像
        for idx, row in gt_df.iterrows():
            # 拼接图像完整路径
            img_path = os.path.join(prefix, row['Filename'])

            # 新增：检查图像文件是否存在
            if not os.path.exists(img_path):
                print(f"警告：图像文件不存在 → {img_path}")
                continue

            # ========== 核心修改：替换cv2.imread为手动解析PPM ==========
            # 原代码：img = cv2.imread(img_path)
            img = read_ppm_manually(img_path)

            if img is not None:  # 避免读取失败的图像
                # 手动解析的PPM已是RGB格式，无需转换（删除原cv2.COLOR_BGR2RGB）
                images.append(img)
                labels.append(row['ClassId'])
    return images, labels


def display_images_and_labels(images, labels):
    """展示每张图片的第一个标签（修正网格布局，适配43类）."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))  # 放大画布，适配更多标签
    i = 1
    # 调整网格为8行6列（8*6=48，足够放下43类）
    grid_rows = 8
    grid_cols = 6
    for label in unique_labels:
        if i > grid_rows * grid_cols:  # 防止超出网格范围
            break
        # 为每个标签选择第一个图片
        image = images[labels.index(label)]
        plt.axis('off')
        plt.subplot(grid_rows, grid_cols, i)
        plt.title(f"标签 {label}（数量：{labels.count(label)}）")
        i += 1
        plt.imshow(image)
    plt.tight_layout()  # 自动调整子图间距
    plt.show()


# ===================== 核心执行逻辑 =====================
# ！！！关键：替换为你本地GTSRB训练集的实际路径！！！
train_image_dir = r"D:\OneDrive\桌面\Traffic_Sign_Classify\GTSRB\Final_Training\Images"

# 新增：验证数据集根路径是否存在
if not os.path.exists(train_image_dir):
    print(f"错误：数据集路径不存在 → {train_image_dir}")
else:
    # 1. 读取图像和标签
    print("正在读取数据集...")
    images, labels = readTrafficSigns(train_image_dir)
    print(f"数据集读取完成：共{len(images)}张图像，{len(set(labels))}类标签")

    # 2. 打印前5张原始图像的信息（仅当读取到图像时执行）
    if images:
        print("\n原始图像信息：")
        for image in images[:5]:
            print(f"形状：{image.shape}, 像素最小值：{image.min()}, 像素最大值：{image.max()}")

        # 3. 替代skimage.transform.resize：用cv2.resize缩放图像到12×12
        print("\n缩放后图像信息：")
        images32 = []
        for image in images[:5]:  # 仅处理前5张，避免耗时
            # cv2.resize参数：(宽, 高)，插值方式选cv2.INTER_AREA（适合缩小图像）
            resized_img = cv2.resize(image, (12, 12), interpolation=cv2.INTER_AREA)
            # 归一化到0-1（和skimage.transform.resize默认行为一致）
            resized_img = resized_img / 255.0
            images32.append(resized_img)
            print(f"形状：{resized_img.shape}, 像素最小值：{resized_img.min():.4f}, 像素最大值：{resized_img.max():.4f}")

        # 4. 展示所有类别图像
        print("\n正在展示图像...")
        display_images_and_labels(images, labels)
    else:
        print("错误：未读取到任何图像，请检查数据集路径或文件完整性！")