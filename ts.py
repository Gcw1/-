import cv2
import numpy as np
from tensorflow import keras
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ===================== 1. 核心配置参数 =====================
MODEL_PATH = r"D:\OneDrive\桌面\Traffic_Sign_Classify\Mymodle\traffic_sign_model.h5"
CLASS_NAMES = [
    "限速20km/h", "限速30km/h", "限速50km/h", "限速60km/h", "限速70km/h",
    "限速80km/h", "解除限速80km/h", "限速100km/h", "限速120km/h", "禁止超车",
    "禁止大型车辆超车", "前方路口", "优先通行", "让行", "停车让行",
    "禁止通行", "禁止大型车辆进入", "禁止驶入", "警告", "急转弯（左）",
    "急转弯（右）", "连续弯道", "路面颠簸", "路面湿滑", "道路变窄（右）",
    "施工", "交通信号灯", "行人", "儿童", "自行车",
    "道路结冰", "野生动物", "解除限速/超车", "右转", "左转",
    "直行", "直行+右转", "直行+左转", "靠右行驶", "靠左行驶",
    "环岛行驶", "解除禁止超车", "解除禁止大型车辆超车"
]
INPUT_SIZE = (32, 32)
# 中文显示所需字体路径（替换为你电脑中的中文字体路径，例如微软雅黑）
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"  # 系统默认微软雅黑路径


# ===================== 2. 工具函数定义 =====================
def read_ppm_manually(file_path):
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


def preprocess_image(img):
    img_resized = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    return img_input


def predict_traffic_sign(model, img_path):
    if img_path.lower().endswith('.ppm'):
        img = read_ppm_manually(img_path)
    else:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        return None, None, None, img

    img_input = preprocess_image(img)
    pred_probs = model.predict(img_input, verbose=0)
    pred_class_id = np.argmax(pred_probs, axis=1)[0]
    pred_confidence = round(pred_probs[0][pred_class_id] * 100, 2)
    pred_class_name = CLASS_NAMES[pred_class_id]

    return pred_class_id, pred_class_name, pred_confidence, img


# ===================== 3. 解决中文显示+文字完整的核心函数 =====================
def draw_chinese_text(img, text, pos=(10, 30), font_size=20, color=(0, 255, 0)):
    """
    使用PIL绘制中文（解决OpenCV不支持中文的问题）
    :param img: BGR格式图像
    :param text: 要显示的文字（含中文）
    :param pos: 文字起始位置
    :param font_size: 字体大小
    :param color: 文字颜色（BGR格式）
    :return: 绘制后的BGR图像
    """
    # 1. 转换为PIL图像（RGB格式）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 2. 加载中文字体
    try:
        font = ImageFont.truetype(FONT_PATH, font_size, encoding="utf-8")
    except:
        print("⚠️ 中文字体加载失败，使用默认字体（可能无法显示中文）")
        font = ImageFont.load_default()

    # 3. 拆分文字为多行（避免超出图像）
    img_w, img_h = pil_img.size
    text_lines = []
    current_line = ""
    for char in text:
        # 估算当前行宽度
        line_width = draw.textlength(current_line + char, font=font)
        if line_width > (img_w - pos[0] - 20):  # 预留边距
            text_lines.append(current_line)
            current_line = char
        else:
            current_line += char
    if current_line:
        text_lines.append(current_line)

    # 4. 逐行绘制文字
    y_offset = pos[1]
    for line in text_lines:
        if y_offset > (img_h - font_size - 10):  # 避免超出图像高度
            break
        draw.text((pos[0], y_offset), line, font=font, fill=(color[2], color[1], color[0]))  # PIL是RGB格式
        y_offset += (font_size + 5)  # 行间距

    # 5. 转换回BGR格式
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr


# ===================== 4. 主函数 =====================
def main():
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    TEST_IMG_PATH = r"D:\OneDrive\桌面\新建文件夹 (4)\1.png"
    class_id, class_name, confidence, img = predict_traffic_sign(model, TEST_IMG_PATH)

    if class_id is not None:
        print("\n📌 单张图像识别结果：")
        print(f"  图像路径：{TEST_IMG_PATH}")
        print(f"  类别ID：{class_id}")
        print(f"  类别名称：{class_name}")
        print(f"  置信度：{confidence}%")

        if img is not None:
            # 调整图像基础尺寸
            img_show = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            base_width = 500
            h, w = img_show.shape[:2]
            scale = base_width / w
            new_h = int(h * scale)
            img_show = cv2.resize(img_show, (base_width, new_h), interpolation=cv2.INTER_CUBIC)

            # 拼接识别文字（含中文）
            text = f"ID:{class_id} | 类别：{class_name} | 置信度：{confidence}%"

            # 绘制中文文字（解决显示问题）
            img_show = draw_chinese_text(
                img_show,
                text,
                pos=(15, 30),
                font_size=22,
                color=(0, 255, 0)
            )

            # 保存高清结果图
            save_path = r"D:\OneDrive\桌面\新建文件夹 (4)\识别结果_中文完整版.png"
            cv2.imwrite(save_path, img_show)
            print(f"📸 结果图已保存至：{save_path}")

            # 显示图像
            cv2.namedWindow("🚦 交通标志识别结果（中文完整）", cv2.WINDOW_NORMAL)
            cv2.imshow("🚦 交通标志识别结果（中文完整）", img_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"\n❌ 图像读取失败：{TEST_IMG_PATH}")


if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()