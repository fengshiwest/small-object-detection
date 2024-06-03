# encoding:utf-8

import os
from tqdm import tqdm
import json
import argparse
import cv2

COCO_Catergories = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor"
}

SCORE_MIN = 0.3


def coco2voc(anno, anno2, xml_dir):
    with open(anno, 'r', encoding='utf-8') as load_f:
        imgs = json.load(load_f)
        imgs = imgs["annotations"]

    with open(anno2, 'r', encoding='utf-8') as load_f1:
        annos = json.load(load_f1)

    for i in tqdm(range(len(imgs))):

        image_id = imgs[i]['image_id']
        image_name = annos['images'][image_id]['file_name']
        image_width = annos['images'][image_id]['width']
        image_height = annos['images'][image_id]['height']

        file_name = image_name.split('.')[0] + ".xml"
        xml_path = os.path.join(xml_dir, file_name)

        bbox = imgs[i]["bbox"]
        category_id = imgs[i]["category_id"]
        # score = imgs[i]["score"]
        score = 1
        cate_name = COCO_Catergories[category_id + 1]

        # if score < SCORE_MIN:
        #     continue

        if not os.path.exists(xml_path):

            xml_content = []

            xml_content.append("<annotation>")
            xml_content.append("	<folder/>")
            xml_content.append("	<filename>" + file_name + "</filename>")
            xml_content.append("    <path></path>")
            xml_content.append("    <source>")
            xml_content.append("        <database>VisDrone</database>")
            xml_content.append("    </source>")
            xml_content.append("	<size>")
            xml_content.append("		<width>" + str(image_width) + "</width>")
            xml_content.append("		<height>" + str(image_height) + "</height>")
            xml_content.append("        <depth>3</depth>")
            xml_content.append("	</size>")
            xml_content.append("	<segmented>0</segmented>")

            xml_content.append("	<object>")
            xml_content.append("		<name>" + cate_name + "</name>")
            xml_content.append("		<pose>Unspecified</pose>")
            xml_content.append("		<truncated>0</truncated>")
            xml_content.append("		<difficult>0</difficult>")
            xml_content.append("		<bndbox>")
            xml_content.append("			<xmin>" + str(int(bbox[0])) + "</xmin>")
            xml_content.append("			<ymin>" + str(int(bbox[1])) + "</ymin>")
            xml_content.append("			<xmax>" + str(int(bbox[0] + bbox[2])) + "</xmax>")
            xml_content.append("			<ymax>" + str(int(bbox[1] + bbox[3])) + "</ymax>")
            xml_content.append("		</bndbox>")
            xml_content.append("        <extra>" + str(score) + "</extra>")
            xml_content.append("	</object>")
            xml_content.append("</annotation>")

            with open(xml_path, 'w+', encoding="utf8") as f:
                f.write('\n'.join(xml_content))
            xml_content[:] = []

        else:

            data = open(xml_path).read()
            xml_content = data.split('\n')

            xml_content.remove("</annotation>")
            xml_content.append("	<object>")
            xml_content.append("		<name>" + cate_name + "</name>")
            xml_content.append("		<pose>Unspecified</pose>")
            xml_content.append("		<truncated>0</truncated>")
            xml_content.append("		<difficult>0</difficult>")
            xml_content.append("		<bndbox>")
            xml_content.append("			<xmin>" + str(int(bbox[0])) + "</xmin>")
            xml_content.append("			<ymin>" + str(int(bbox[1])) + "</ymin>")
            xml_content.append("			<xmax>" + str(int(bbox[0] + bbox[2])) + "</xmax>")
            xml_content.append("			<ymax>" + str(int(bbox[1] + bbox[3])) + "</ymax>")
            xml_content.append("		</bndbox>")
            xml_content.append("        <extra>" + str(score) + "</extra>")
            xml_content.append("	</object>")
            xml_content.append("</annotation>")

            with open(xml_path, 'w+', encoding="utf8") as f:
                f.write('\n'.join(xml_content))
            xml_content[:] = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert coco .json annotation to voc .xml annotation")
    parser.add_argument('--json_path', type=str, help='path to json file.', default="data/VisDrone/annotations/instance_UFP_gfocal_r50_fpn_1x_ciou_pretrained_dense_MCNN_UAVval.json")
    parser.add_argument('--anno_path', type=str, help='path to json file.', default="data/VisDrone/annotations/UFPMP_val.json")
    parser.add_argument('--output', type=str, help='path to output xml files.', default="result/gfocal_r50_fpn_1x_ciou_pretrained_dense_MCNN_val")
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    coco2voc(args.json_path, args.anno_path, args.output)
