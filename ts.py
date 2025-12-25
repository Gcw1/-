import cv2
import numpy as np
from tensorflow import keras
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import math

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
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"

# ===================== 批量处理+指标配置 =====================
INPUT_FOLDER = r"D:\OneDrive\桌面\新建文件夹 (4)"
OUTPUT_FOLDER = r"D:\OneDrive\桌面\新建文件夹 (4)\识别结果"
SUPPORTED_FORMATS = ('.ppm', '.png', '.jpg', '.jpeg', '.bmp')

# 指标配置
LOW_CONFIDENCE_THRESHOLD = 80.0  # 低置信度阈值（低于此值标记预警）
REPORT_SAVE_PATH = os.path.join(OUTPUT_FOLDER, "批量识别指标报告.csv")
METRICS_TXT_PATH = os.path.join(OUTPUT_FOLDER, "批量识别指标汇总.txt")
PLOT_SAVE_PATH = os.path.join(OUTPUT_FOLDER, "识别指标可视化.png")

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


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
        if img is not None:
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
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(FONT_PATH, font_size, encoding="utf-8")
    except:
        print(" 中文字体加载失败，使用默认字体（可能无法显示中文）")
        font = ImageFont.load_default()

    img_w, img_h = pil_img.size
    text_lines = []
    current_line = ""
    for char in text:
        line_width = draw.textlength(current_line + char, font=font)
        if line_width > (img_w - pos[0] - 20):
            text_lines.append(current_line)
            current_line = char
        else:
            current_line += char
    if current_line:
        text_lines.append(current_line)

    y_offset = pos[1]
    for line in text_lines:
        if y_offset > (img_h - font_size - 10):
            break
        draw.text((pos[0], y_offset), line, font=font, fill=(color[2], color[1], color[0]))
        y_offset += (font_size + 5)

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr


# ===================== 4. 新增：指标计算与可视化函数 =====================
def calculate_metrics(result_data):
    """
    计算识别指标
    :param result_data: 列表，每个元素为 [文件名, 类别ID, 类别名称, 置信度]
    :return: 指标字典 + 详细数据DataFrame
    """
    df = pd.DataFrame(result_data, columns=['文件名', '类别ID', '类别名称', '置信度'])

    # 基础统计指标
    total_images = len(df)
    confidence_scores = df['置信度'].tolist()

    # 置信度统计
    conf_mean = round(np.mean(confidence_scores), 2)
    conf_min = round(np.min(confidence_scores), 2)
    conf_max = round(np.max(confidence_scores), 2)
    conf_std = round(np.std(confidence_scores), 2)
    conf_median = round(np.median(confidence_scores), 2)

    # 低置信度预警
    low_conf_count = len(df[df['置信度'] < LOW_CONFIDENCE_THRESHOLD])
    low_conf_ratio = round(low_conf_count / total_images * 100, 2)
    low_conf_files = df[df['置信度'] < LOW_CONFIDENCE_THRESHOLD]['文件名'].tolist()

    # 类别分布统计
    class_distribution = Counter(df['类别名称'])
    top_class = max(class_distribution, key=class_distribution.get) if class_distribution else None
    top_class_count = class_distribution.get(top_class, 0)

    # 封装指标字典
    metrics = {
        '总识别图像数': total_images,
        '置信度平均值': conf_mean,
        '置信度最小值': conf_min,
        '置信度最大值': conf_max,
        '置信度标准差': conf_std,
        '置信度中位数': conf_median,
        '低置信度阈值': LOW_CONFIDENCE_THRESHOLD,
        '低置信度图像数': low_conf_count,
        '低置信度占比(%)': low_conf_ratio,
        '识别最多的类别': top_class,
        '识别最多类别的数量': top_class_count,
        '识别类别总数': len(class_distribution)
    }

    return metrics, df, low_conf_files


def plot_metrics_visualization(metrics, df):
    """绘制指标可视化图表"""
    fig = plt.figure(figsize=(16, 10))

    # 子图1：置信度分布直方图
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(df['置信度'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    ax1.axvline(metrics['置信度平均值'], color='red', linestyle='--', label=f'平均值：{metrics["置信度平均值"]}%')
    ax1.axvline(LOW_CONFIDENCE_THRESHOLD, color='orange', linestyle='--',
                label=f'低置信阈值：{LOW_CONFIDENCE_THRESHOLD}%')
    ax1.set_xlabel('识别置信度（%）')
    ax1.set_ylabel('图像数量')
    ax1.set_title('置信度分布')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 子图2：类别识别数量TOP10
    ax2 = plt.subplot(2, 2, 2)
    class_dist = Counter(df['类别名称'])
    top10_classes = dict(sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[:10])
    ax2.barh(list(top10_classes.keys()), list(top10_classes.values()), color='skyblue')
    ax2.set_xlabel('识别数量')
    ax2.set_title('识别数量TOP10类别')
    ax2.grid(axis='x', alpha=0.3)

    # 子图3：关键指标汇总（文本）
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    metrics_text = f"""
    批量识别核心指标汇总
    --------------------
    总识别图像数：{metrics['总识别图像数']}
    置信度统计：
      平均值：{metrics['置信度平均值']}%
      最小值：{metrics['置信度最小值']}%
      最大值：{metrics['置信度最大值']}%
      标准差：{metrics['置信度标准差']}%
      中位数：{metrics['置信度中位数']}%
    低置信度预警：
      阈值：{metrics['低置信度阈值']}%
      数量：{metrics['低置信度图像数']}
      占比：{metrics['低置信度占比(%)']}%
    类别分布：
      识别类别总数：{metrics['识别类别总数']}
      识别最多类别：{metrics['识别最多的类别']}（{metrics['识别最多类别的数量']}张）
    """
    ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # 子图4：低置信度占比饼图
    ax4 = plt.subplot(2, 2, 4)
    low_conf = metrics['低置信度图像数']
    high_conf = metrics['总识别图像数'] - low_conf
    labels = [f'高置信度（≥{LOW_CONFIDENCE_THRESHOLD}%）', f'低置信度（<{LOW_CONFIDENCE_THRESHOLD}%）']
    sizes = [high_conf, low_conf]
    colors = ['#66b3ff', '#ff9999']
    explode = (0, 0.1)  # 突出低置信度部分
    ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax4.set_title('置信度等级分布')

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150, bbox_inches='tight')
    print(f"\n 指标可视化图表已保存至：{PLOT_SAVE_PATH}")
    plt.show()


def save_metrics_report(metrics, df, low_conf_files):
    """保存指标报告（CSV+文本）"""
    # 保存详细数据CSV
    df['低置信度预警'] = df['置信度'] < LOW_CONFIDENCE_THRESHOLD
    df.to_csv(REPORT_SAVE_PATH, index=False, encoding='utf-8-sig')
    print(f" 详细识别报告已保存至：{REPORT_SAVE_PATH}")

    # 保存指标汇总文本
    with open(METRICS_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("批量交通标志识别指标汇总报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. 基础统计指标\n")
        f.write(f"   总识别图像数：{metrics['总识别图像数']}\n")
        f.write(f"   识别成功数：{metrics['总识别图像数']}\n")

        f.write("\n2. 置信度统计指标\n")
        f.write(f"   平均值：{metrics['置信度平均值']}%\n")
        f.write(f"   最小值：{metrics['置信度最小值']}%\n")
        f.write(f"   最大值：{metrics['置信度最大值']}%\n")
        f.write(f"   标准差：{metrics['置信度标准差']}%\n")
        f.write(f"   中位数：{metrics['置信度中位数']}%\n")

        f.write("\n3. 低置信度预警指标\n")
        f.write(f"   预警阈值：{metrics['低置信度阈值']}%\n")
        f.write(f"   预警数量：{metrics['低置信度图像数']}\n")
        f.write(f"   预警占比：{metrics['低置信度占比(%)']}%\n")
        if low_conf_files:
            f.write(f"   预警文件列表：{', '.join(low_conf_files)}\n")

        f.write("\n4. 类别分布指标\n")
        f.write(f"   识别类别总数：{metrics['识别类别总数']}\n")
        f.write(f"   识别最多的类别：{metrics['识别最多的类别']}（{metrics['识别最多类别的数量']}张）\n")

        f.write("\n5. 质量评估结论\n")
        if metrics['低置信度占比(%)'] < 10:
            f.write("    识别质量优秀：低置信度图像占比低于10%\n")
        elif metrics['低置信度占比(%)'] < 20:
            f.write("     识别质量良好：低置信度图像占比10%-20%\n")
        else:
            f.write("    识别质量较差：低置信度图像占比高于20%，建议检查模型或图像质量\n")

    print(f" 指标汇总报告已保存至：{METRICS_TXT_PATH}")


# ===================== 5. 批量处理核心函数（新增指标计算） =====================
def batch_process(model):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    total_count = 0
    success_count = 0
    fail_list = []
    result_data = []  # 存储识别结果用于指标计算

    # 遍历文件
    file_list = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(SUPPORTED_FORMATS)]
    if not file_list:
        print("  未找到支持的图像文件！")
        return

    print(f"\n 开始批量处理 {len(file_list)} 个文件...")

    for file_name in file_list:
        total_count += 1
        img_path = os.path.join(INPUT_FOLDER, file_name)
        print(f"\n正在处理：{file_name}")

        class_id, class_name, confidence, img = predict_traffic_sign(model, img_path)

        if class_id is not None and img is not None:
            success_count += 1
            result_data.append([file_name, class_id, class_name, confidence])

            # 打印单张识别结果
            print(f"  类别ID：{class_id}")
            print(f"  类别名称：{class_name}")
            print(f"  置信度：{confidence}%")
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                print(f"    低置信度预警！（低于{LOW_CONFIDENCE_THRESHOLD}%）")

            # 处理并保存图像
            img_show = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            base_width = 500
            h, w = img_show.shape[:2]
            scale = base_width / w
            new_h = int(h * scale)
            img_show = cv2.resize(img_show, (base_width, new_h), interpolation=cv2.INTER_CUBIC)

            # 拼接文字（新增低置信度预警标识）
            warn_tag = " 低置信度 " if confidence < LOW_CONFIDENCE_THRESHOLD else ""
            text = f"{warn_tag}文件名：{file_name} | ID:{class_id} | 类别：{class_name} | 置信度：{confidence}%"

            # 低置信度文字用红色标注
            text_color = (0, 0, 255) if confidence < LOW_CONFIDENCE_THRESHOLD else (0, 255, 0)
            img_show = draw_chinese_text(
                img_show,
                text,
                pos=(15, 30),
                font_size=22,
                color=text_color
            )

            save_name = f"{os.path.splitext(file_name)[0]}_识别结果.png"
            save_path = os.path.join(OUTPUT_FOLDER, save_name)
            cv2.imwrite(save_path, img_show)
            print(f"  结果图已保存至：{save_path}")
        else:
            fail_list.append(file_name)
            print(f"  处理失败：{file_name}")

    # 输出基础统计
    print("\n" + "=" * 60)
    print("批量处理完成！基础统计结果：")
    print(f"总扫描文件数：{total_count}")
    print(f"成功识别数：{success_count}")
    print(f"处理失败数：{len(fail_list)}")
    if fail_list:
        print(f"失败文件列表：{fail_list}")
    print("=" * 60)

    # 计算并保存详细指标
    if result_data:
        print("\n 开始计算识别指标...")
        metrics, df, low_conf_files = calculate_metrics(result_data)

        # 输出关键指标
        print("\n" + "=" * 60)
        print("核心识别指标汇总：")
        print(f"置信度平均值：{metrics['置信度平均值']}%")
        print(f"低置信度图像数：{metrics['低置信度图像数']}（占比{metrics['低置信度占比(%)']}%）")
        print(f"识别类别总数：{metrics['识别类别总数']}")
        print("=" * 60)

        # 保存报告
        save_metrics_report(metrics, df, low_conf_files)

        # 绘制可视化图表
        plot_metrics_visualization(metrics, df)

    # 显示示例图像
    if success_count > 0:
        success_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('_识别结果.png')]
        if success_files:
            last_file = success_files[-1]
            last_path = os.path.join(OUTPUT_FOLDER, last_file)
            last_img = cv2.imread(last_path)
            if last_img is not None:
                cv2.namedWindow("🚦 批量处理示例结果（按任意键关闭）", cv2.WINDOW_NORMAL)
                cv2.imshow("🚦 批量处理示例结果（按任意键关闭）", last_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


# ===================== 6. 主函数 =====================
def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 加载模型
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(" 模型加载成功！")
    except Exception as e:
        print(f" 模型加载失败：{e}")
        return

    # 执行批量处理
    print(f"\n 待处理文件夹：{INPUT_FOLDER}")
    print(f" 支持格式：{SUPPORTED_FORMATS}")
    print(f" 低置信度预警阈值：{LOW_CONFIDENCE_THRESHOLD}%")
    batch_process(model)


if __name__ == "__main__":
    # 安装依赖（首次运行取消注释）
    # os.system("pip install pandas matplotlib numpy")
    main()