# -*- coding: utf-8 -*-
import argparse
import os
import sys
import cv2
import numpy as np
import torch
import xml.dom.minidom
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.etree.ElementTree import ElementTree

sys.path.append(os.path.dirname(os.getcwd()))

from train.model.UNet.unet_model import UNet
from CRAFT.predict import TextDetector
from train.utils.dataset import BasicDataset
from train.utils.utils import get_polygon_from_lines, is_pt_in_polygon


def convert_str_coord_to_int(str_coord):
    pts_str = str_coord.split(" ")
    pts = []
    for i in range(len(pts_str)):
        x, y = pts_str[i].split(",")
        pts.append([int(x), int(y)])

    return pts


def convert_pt_to_string(pts):
    pt_str = ""
    for pt in pts:
        if pt_str != "":
            pt_str += " "
        pt_str += str(int(pt[0])) + "," + str(int(pt[1]))

    return pt_str


def get_text_box_in_cell(cell, texts, text_status):
    inside_text = []
    for i in range(len(texts)):
        if not text_status[i]:
            center = np.mean(texts[i], axis=0)
            if is_pt_in_polygon(center, cell) >= 0:
                inside_text.append(texts[i])
                text_status[i] = True

    box = []
    if inside_text:
        minRect = cv2.minAreaRect(np.array(inside_text).reshape(-1, 2))
        box = cv2.boxPoints(minRect).astype(np.int)

    return box


class Predictor:
    def __init__(self, pretrained_model_path, text_model_path):
        self.img_size = (640, 640)

        self.net = UNet(n_channels=3, n_classes=2, bilinear=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net.to(device=self.device)
        self.net.load_state_dict(torch.load(pretrained_model_path, map_location=self.device))
        print('Loading weights from checkpoint ' + pretrained_model_path)

        self.out_threshold = 0.7
        self.text_detector = TextDetector(text_model_path)

    def get_masks(self, full_img):
        self.net.eval()

        img, scale_rate = BasicDataset.preprocess(full_img, self.img_size)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.net(img)

            probs = torch.sigmoid(output)
            probs = probs.squeeze(0)
            masks = probs.cpu().numpy()
            masks = masks.transpose((1, 2, 0))

        # 二值化
        h, w = full_img.shape[:2]
        img_area = (int(round(w * scale_rate[0])), int(round(h * scale_rate[1])))
        masks = masks[: img_area[1], : img_area[0], :]
        masks = cv2.resize(masks, (w, h))
        masks = masks.transpose((2, 0, 1))

        masks = np.where(masks > self.out_threshold, 255, 0)

        return masks

    def get_cells_from_masks(self, masks):
        h, w = masks[0].shape[:2]
        area_thresh = h * w * 0.001

        lines = [[], []]  # 分别表示横线和竖线
        for i in range(len(masks)):
            mask = masks[i].astype("uint8")
            _, contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                Mo = cv2.moments(contour)  # 求矩
                area = Mo["m00"]

                if area > area_thresh:
                    dim = 0 if i % 2 == 0 else 1
                    c_sort = sorted(contour.reshape(-1, 2), key=lambda x: x[dim])
                    line = [c_sort[0].tolist(), c_sort[-1].tolist()]

                    lines[dim].append(line)

        # 排序
        lines[0] = sorted(lines[0], key=lambda x: x[0][1])
        lines[1] = sorted(lines[1], key=lambda x: x[0][0])

        cells = []
        for i in range(len(lines[0]) - 1):
            for j in range(len(lines[1]) - 1):
                pts = get_polygon_from_lines([lines[1][j], lines[0][i], lines[1][j + 1], lines[0][i + 1]])
                cell = {"begin_row": i, "end_row": i, "begin_col": j, "end_col": j, "pts": pts}
                cells.append(cell)

        return cells

    def add_cell_to_result(self, table_xml, cells):
        for cell in cells:
            cell_xml = SubElement(table_xml, "cell")
            cell_xml.set("end-col", str(cell["end_col"]))
            cell_xml.set("end-row", str(cell["end_row"]))
            cell_xml.set("start-col", str(cell["begin_col"]))
            cell_xml.set("start-row", str(cell["begin_row"]))
            cell_coord = SubElement(cell_xml, "Coords")
            cell_coord.set("points", convert_pt_to_string(cell["pts"]))

    def predict(self, input_path, output_path):
        files = os.listdir(input_path + "/imgs")
        for file in files:
            name, type = file.split(".")
            if type == "xml":
                continue

            texts = self.text_detector.predict(os.path.join(input_path + "/imgs", file))
            text_status = [False] * len(texts)

            img = cv2.imread(os.path.join(input_path + "/imgs", file))

            gt_dom = xml.dom.minidom.parse(os.path.join(input_path + "/gt", name + ".xml"))
            gt_tables = gt_dom.documentElement.getElementsByTagName("table")

            root = Element("document")
            root.set("imgname", name)

            for i in range(len(gt_tables)):
                gt_table = gt_tables[i]
                table_coord_str = gt_table.firstChild.getAttribute("points")
                table_pts = convert_str_coord_to_int(table_coord_str)

                c_min = np.min(table_pts, axis=0)
                c_max = np.max(table_pts, axis=0)

                table_img = img[c_min[1]: c_max[1], c_min[0]: c_max[0]]
                masks = self.get_masks(table_img)

                cells = self.get_cells_from_masks(masks)

                offset = [c_min[0], c_min[1]]  # x和y的offset
                valid_cell = []
                for cell in cells:
                    for j in range(4):
                        cell["pts"][j] = [cell["pts"][j][0] + offset[0], cell["pts"][j][1] + offset[1]]

                    text_box = get_text_box_in_cell(cell["pts"], texts, text_status)
                    if text_box != []:
                        cell["pts"] = text_box
                        valid_cell.append(cell)

                table = SubElement(root, "table")
                table.set("id", str(i + 1))
                table_coord = SubElement(table, "Coords")
                table_coord.set("points", table_coord_str)

                self.add_cell_to_result(table, valid_cell)

            # 将xml文件保存在输出文件夹中
            xml_tree = ElementTree(root)
            xml_tree.write(os.path.join(output_path, name + ".xml"), encoding="utf-8")


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", dest="input", help="input file path", required=True)
    parser.add_argument("-o", "--output", dest="output", help="output file directory", required=True)
    parser.add_argument("-p", "--model_path", dest="model", help="path of model", required=True)
    parser.add_argument("-t", "--text_model_path", dest="text_model", help="path of model", required=True)

    args = parser.parse_args()

    predictor = Predictor(args.model, args.text_model)
    predictor.predict(args.input, args.output)
    print("done")


if __name__ == "__main__":
    run()
