import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import cv2
import xml.dom.minidom


def convert_string_to_pt(string):
    items = string.split(' ')
    pts = []

    for item in items:
        loc = item.split(',')
        pt = [int(loc[0]), int(loc[1])]
        pts.append(pt)
    return pts


def get_row_col_nb(row_col_cells):
    nb = [0, 0]
    for i in range(2):
        for j in row_col_cells[i].keys():
            if int(j) + 1 > nb[i]:
                nb[i] = (int(j) + 1)

    return nb


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, dst_size, mask_suffix=""):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.dst_size = dst_size
        self.mask_suffix = mask_suffix

        # self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir) if not file.startswith(".")]
        self.file_nb = 0
        self.imgs = []
        for file in os.listdir(imgs_dir):
            name, type = file.split('.')
            if type == 'xml':
                continue

            self.file_nb += 1

            gt_dom = xml.dom.minidom.parse(os.path.join(masks_dir, name + '.xml'))
            gt_tables = gt_dom.documentElement.getElementsByTagName("table")

            for i in range(len(gt_tables)):
                gt_table = gt_tables[i]
                nodes = gt_table.childNodes

                table_coord_str = nodes[0].getAttribute('points')
                table_pts = convert_string_to_pt(table_coord_str)

                cell_infos = []
                for j in range(1, len(nodes)):
                    info_temp = {'row_col': [nodes[j].getAttribute('start-row'), nodes[j].getAttribute('end-row'),
                                             nodes[j].getAttribute('start-col'), nodes[j].getAttribute('end-col')],
                                 "points": convert_string_to_pt(nodes[j].firstChild.getAttribute('points'))}
                    cell_infos.append(info_temp)

                self.imgs.append({'img_name': file, 'table_loc': table_pts, 'cell_infos': cell_infos})

        logging.info(f"Creating dataset with {len(self.imgs)} examples")

    def __len__(self):
        return self.file_nb

    @classmethod
    def preprocess(cls, img, dstsize, normalization=True):
        img_nd = img
        h, w = img.shape[:2]

        rate_x = min(dstsize[0] / w, 1.0)
        rate_y = min(dstsize[1] / h, 1.0)
        img_nd = cv2.resize(img_nd, None, fx=rate_x, fy=rate_y, interpolation=cv2.INTER_AREA)
        h, w = img_nd.shape[:2]
        pad_x = dstsize[0] - w
        pad_y = dstsize[1] - h

        if len(img.shape) == 2:
            img_nd = cv2.threshold(img_nd, 200, 255, cv2.THRESH_BINARY)[1]
            img_nd = np.expand_dims(img_nd, axis=2)

        img_nd = np.pad(img_nd, ((0, pad_y), (0, pad_x), (0, 0)), mode="constant")

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1 and normalization:
            img_trans = img_trans / 255

        return img_trans, (rate_x, rate_y)

    def get_mask(self, cell_infos, img_area):
        offset = np.array(img_area[:2])
        w, h = img_area[2], img_area[3]
        masks = [np.zeros((h, w)), np.zeros((h, w))]

        # draw mask from label
        row_col_cells = [{}, {}]
        all_pts = []
        for cell_info in cell_infos:
            idxs = cell_info['row_col']
            pts = np.array(cell_info['points'])
            pts -= offset

            all_pts.append(pts)
            if idxs[0] == idxs[1] and idxs[2] == idxs[3]:
                row_col_idx = [idxs[0], idxs[2]]

                for m in range(2):
                    if row_col_idx[m] in row_col_cells[m].keys():
                        row_col_cells[m][row_col_idx[m]].append(pts)
                    else:
                        row_col_cells[m][row_col_idx[m]] = [pts]

        table_box = cv2.boundingRect(np.array(all_pts).reshape(-1, 2))

        row_col_nb = get_row_col_nb(row_col_cells)
        for i in range(2):
            line_cells = row_col_cells[i]
            for j in range(row_col_nb[i] - 1):
                if str(j) in line_cells.keys() and str(j + 1) in line_cells.keys():
                    up = cv2.boundingRect(np.array(line_cells[str(j)]).reshape(-1, 2))
                    down = cv2.boundingRect(np.array(line_cells[str(j + 1)]).reshape(-1, 2))

                    if i == 0:
                        cv2.rectangle(masks[i], (table_box[0], up[1] + up[3]), (table_box[0] + table_box[2], down[1]), 255, thickness=-1)
                    else:
                        cv2.rectangle(masks[i], (up[0] + up[2], table_box[1]), (down[0], table_box[1] + table_box[3]), 255, thickness=-1)

        return masks

    def __getitem__(self, i):
        img_infos = self.imgs[i]
        img_file = img_infos['img_name']

        img = cv2.imread(os.path.join(self.imgs_dir, img_file))
        img_area = cv2.boundingRect(np.array(img_infos['table_loc']))
        img_table = img[img_area[1]: img_area[1] + img_area[3], img_area[0]: img_area[0] + img_area[2], :]

        masks = self.get_mask(img_infos['cell_infos'], img_area)

        img_table, _ = self.preprocess(img_table, self.dst_size)

        mask_array = []
        for i in range(len(masks)):
            mask_temp, _ = self.preprocess(masks[i], self.dst_size)
            mask_array.append(mask_temp[0, :, :])

        mask_array = np.array(mask_array)

        assert (
            img_table.shape[1:3] == mask_array[0].shape[0:2]
        ), f"Image and mask {img_file} should be the same size, but are {img_table.shape} and {mask_array[0].shape}"

        return {
            "image": torch.from_numpy(img_table).type(torch.FloatTensor),
            "mask": torch.from_numpy(mask_array).type(torch.FloatTensor),
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix="_mask")
