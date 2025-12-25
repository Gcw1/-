import numpy as np
import os
import pandas as pd

def readTrafficSigns(rootpath):
    images = []
    labels = []
    for c in range(43):
        prefix = os.path.join(rootpath, format(c, '05d'))
        gt_file = os.path.join(prefix, 'GT-' + format(c, '05d') + '.csv')
        if not os.path.exists(gt_file):
            continue
        gt_df = pd.read_csv(gt_file, sep=';')
        for idx, row in gt_df.iterrows():
            img_path = os.path.join(prefix, row['Filename'])
            # 用手动解析PPM
            def read_ppm(file_path):
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
                except:
                    return None
            img = read_ppm(img_path)
            if img is not None:
                images.append(img)
                labels.append(row['ClassId'])
    return images, labels


if __name__ == '__main__':
    # 加r前缀，修正\F转义警告
    readTrafficSigns(r'GTSRB\Final_Training\Images')