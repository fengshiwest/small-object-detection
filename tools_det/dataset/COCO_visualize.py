import json
import os
import cv2
import numpy as np


def generate_colors(num_classes):
    # 生成 num_classes 个不同的颜色
    colors = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    return colors.tolist()


# 确认图像和标签数量
def val(json_path, outpath, image_path):
    # 打开 JSON 文件并加载数据
    json_file = open(json_path)
    infos = json.load(json_file)

    # 获取图像和标注信息
    images = infos["images"]
    annos = infos["annotations"]

    print('图像数量为：', len(images))
    print('标签数量为：', len(annos))


# 可视化边界框
def select(json_path, outpath, image_path):
    # 打开 JSON 文件并加载数据
    json_file = open(json_path)
    infos = json.load(json_file)

    # 获取图像和标注信息
    images = infos["images"]
    annos = infos["annotations"]
    categories = infos["categories"]

    # 获取类别数和生成颜色
    num_classes = len(categories)
    colors = generate_colors(num_classes)

    # 遍历每张图像
    for i in range(len(images)):
        im_id = images[i]["id"]
        im_path = os.path.join(image_path, images[i]["file_name"])

        # 读取图像
        img = cv2.imread(im_path)

        # 遍历每个标注信息
        for j in range(len(annos)):
            if annos[j]["image_id"] == im_id:
                x, y, w, h = annos[j]["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)
                x2, y2 = x + w, y + h

                # 获取类别信息和颜色
                category_id = annos[j]["category_id"]
                category_info = next((cat for cat in categories if cat["id"] == category_id), None)
                category_name = category_info["name"] if category_info else "Unknown"
                color = colors[category_id % num_classes]

                # 在图像上绘制边界框和类别信息
                label = f"{category_name}"
                img = cv2.rectangle(img, (x, y), (x2, y2), color, thickness=2)
                img = cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 保存结果图像
                img_name = os.path.join(outpath, images[i]["file_name"])
                cv2.imwrite(img_name, img)

        # 打印当前处理的图像序号
        print(i + 1)


if __name__ == "__main__":
    # 输入文件路径和输出路径
    json_path = "data/COCO2017/annotations/instances_val2017.json"  # 放标注json的地址
    out_path = "data/COCO2017/val2017_vis"  # 结果放的地址
    image_path = "data/COCO2017/val2017"  # 原图的地址

    # 调用可视化函数进行处理
    select(json_path, out_path, image_path)
    # 调用验证函数进行处理
    # val(json_path, out_path, image_path)