from eval import eval
from data_structure import Table
import os
import xml.dom.minidom
from os.path import join as osj
import argparse
import json
import traceback


# calculate the gt adj_relations of the missing file
# @param: file_lst - list of missing ground truth file
# @param: cur_gt_num - current total of ground truth objects (tables / cells)
def process_missing_files(gt_path, file_lst, cur_gt_num):
    gt_file_lst_full = [osj(gt_path, filename) for filename in file_lst]
    for file in gt_file_lst_full:
        if os.path.split(file)[-1].split(".")[-1] == "xml":
            gt_dom = xml.dom.minidom.parse(file)
            gt_root = gt_dom.documentElement
            tables = []
            table_elements = gt_root.getElementsByTagName("table")
            for res_table in table_elements:
                t = Table(res_table)
                tables.append(t)
            for table in tables:
                cur_gt_num += len(table.find_adj_relations())

    return cur_gt_num


def calc_mean_F1(gt_path, result, iou_list):
    gt_file_lst = os.listdir(gt_path)
    # note: results are stored as list of each when iou at [0.5, 0.6, gt_filename]
    iou_nb = len(iou_list)
    correct_list, res_list = [0 for i in range(iou_nb)], [0 for i in range(iou_nb)]

    gt_num = 0
    for each_file in result:
        gt_file_lst.remove(each_file.result[-1])
        for i in range(iou_nb):
            correct_list[i] += each_file.result[i].truePos
            res_list[i] += each_file.result[i].resTotal

        gt_num += each_file.result[0].gtTotal

    for file in gt_file_lst:
        if file.split(".")[-1] != "xml":
            gt_file_lst.remove(file)

    if len(gt_file_lst) > 0:
        # print("\nWarning: missing result annotations for file: {}\n".format(gt_file_lst))
        gt_total = process_missing_files(gt_path, gt_file_lst, gt_num)
        # gt_total = gt_num
    else:
        gt_total = gt_num

    meanF1 = 0
    for i in range(iou_nb):
        try:
            precision = correct_list[i] / res_list[i]
            recall = correct_list[i] / gt_total
            F1 = 2 * precision * recall / (precision + recall)

            print("IOU @ {} -\nprecision: {}\nrecall: {}\nf1: {}".format(iou_list[i], precision, recall, F1))
            print("correct: {}, gt: {}, res: {}".format(correct_list[i], gt_total, res_list[i]))
        except ZeroDivisionError:
            print(
                "Error: zero devision error found, (possible that no adjacency relations are found), please check the "
                "file input. "
            )
            F1 = 0

        meanF1 += F1

    meanF1 /= iou_nb

    return meanF1


def get_args():
    parser = argparse.ArgumentParser(
        description="Eval result files", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-r", "--result_path", required=True, type=str, dest="result", help="path of result files")
    parser.add_argument("-gt", "--gt_path", required=True, type=str, dest="gt", help="path of gt files")
    parser.add_argument("-o", "--output_path", required=True, type=str, dest="output", help="path of gt files")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    try:
        result_files = os.listdir(args.result)
        print("number of result files:", len(result_files))
        result_path = args.result
        cell_ious = [0.5, 0.6]  # 映射单元格时的iou阈值

        print("start to process files")
        res_lst = []
        count = 0

        for file in result_files:
            name, type = file.split(".")
            if type == "xml":
                if count % 10 == 0:
                    print("count", count)

                cur_filepath = osj(os.path.abspath(args.result), file)
                res = eval(cur_filepath, args.gt, cell_ious=cell_ious)
                res_lst.append(res)
                count += 1

                # print("Processing... {}".format(name))

        print("start to calc meanF1")

        meanF1 = calc_mean_F1(args.gt, res_lst, cell_ious)
        print("mean F1:", meanF1)
        eval_result = {"meanF1": round(meanF1, 5)}
        json_str = json.dumps(eval_result, indent=1)
        with open(args.output, "w") as f:
            f.write(json_str)

    except Exception:
        print(traceback.format_exc())
        eval_result = {"meanF1": -1}
        json_str = json.dumps(eval_result, indent=1)
        with open(args.output, "w") as f:
            f.write(json_str)

    print("done")
