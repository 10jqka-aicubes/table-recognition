import os
import sys
import xml.dom.minidom
from data_structure import ResultStructure, Table, AdjRelation
from os.path import join as osj


class eval:
    STR = "-str"
    REG = "-reg"
    DEFAULT_ENCODING = "UTF-8"

    def __init__(self, res_path, gt_path, cell_ious=None):
        if cell_ious is None:
            cell_ious = [0.5, 0.6]
        self.cell_ious = cell_ious

        self.return_result = None
        self.reg = False
        self.str = False

        self.resultFile = res_path
        self.inPrefix = os.path.split(res_path)[-1].split(".")[0]

        self.str = True
        self.GTFile = osj(gt_path, self.inPrefix + ".xml")

        self.gene_ret_lst()

    @property
    def result(self):
        return self.return_result

    def gene_ret_lst(self):
        ret_lst = []
        for iou in self.cell_ious:
            temp = self.compute_retVal(iou)
            ret_lst.append(temp)

        ret_lst.append(self.inPrefix + ".xml")
        # print("Done processing {}\n".format(self.resultFile))
        self.return_result = ret_lst

    def compute_retVal(self, iou):
        gt_dom = xml.dom.minidom.parse(self.GTFile)
        # incorrect submission format handling
        try:
            result_dom = xml.dom.minidom.parse(self.resultFile)
        except Exception:
            # result_dom = xml.dom.minidom.parse(dummyDom)
            gt_tables = eval.get_table_list(gt_dom)
            retVal = ResultStructure(truePos=0, gtTotal=len(gt_tables), resTotal=0)
            return retVal

        # result_dom = xml.dom.minidom.parse(self.resultFile)
        if self.reg:
            ret = self.evaluate_result_reg(gt_dom, result_dom, iou)
            return ret
        if self.str:
            ret = self.evaluate_result_str(gt_dom, result_dom, iou)
            return ret

    @staticmethod
    def get_table_list(dom):
        """
        return a list of Table objects corresponding to the table element of the DOM.
        """
        return [Table(_nd) for _nd in dom.documentElement.getElementsByTagName("table")]

    @staticmethod
    def evaluate_result_reg(gt_dom, result_dom, iou_value):
        # parse the tables in input elements
        gt_tables = eval.get_table_list(gt_dom)
        result_tables = eval.get_table_list(result_dom)

        # duplicate result table list
        remaining_tables = result_tables.copy()

        # map the tables in gt and result file
        table_matches = []  # @param: table_matches - list of mapping of tables in gt and res file, in order (gt, res)
        for gtt in gt_tables:
            for rest in remaining_tables:
                if gtt.compute_table_iou(rest) >= iou_value:
                    remaining_tables.remove(rest)
                    table_matches.append((gtt, rest))
                    break

        assert len(table_matches) <= len(gt_tables)
        assert len(table_matches) <= len(result_tables)

        retVal = ResultStructure(truePos=len(table_matches), gtTotal=len(gt_tables), resTotal=len(result_tables))
        return retVal

    @staticmethod
    def evaluate_result_str(gt_dom, result_dom, iou_value, table_iou_value=0.8):
        # parse the tables in input elements
        gt_tables = eval.get_table_list(gt_dom)
        result_tables = eval.get_table_list(result_dom)

        gt_match_status = [False for table in gt_tables]
        rest_match_status = [False for table in result_tables]
        table_matches = []  # @param: table_matches - list of mapping of tables in gt and res file, in order (gt, res)
        for i in range(len(gt_tables)):
            gtt = gt_tables[i]
            for j in range(len(result_tables)):
                if rest_match_status[j]:
                    continue
                rest = result_tables[j]
                # note: for structural analysis, use 0.8 for table mapping
                if gtt.compute_table_iou(rest) >= table_iou_value:
                    table_matches.append((gtt, rest))
                    gt_match_status[i] = True
                    rest_match_status[j] = True
                    break

        gt_remaining, remaining_tables = [], []
        for i in range(len(gt_match_status)):
            if not gt_match_status[i]:
                gt_remaining.append(gt_tables[i])

        for i in range(len(rest_match_status)):
            if not rest_match_status[i]:
                remaining_tables.append(result_tables[i])

        total_gt_relation, total_res_relation, total_correct_relation = 0, 0, 0
        for gt_table, ress_table in table_matches:
            # set up the cell mapping for matching tables
            cell_mapping = gt_table.find_cell_mapping(ress_table, iou_value)
            # set up the adj relations, convert the one for result table to a dictionary for faster searching
            gt_AR = gt_table.find_adj_relations()
            total_gt_relation += len(gt_AR)

            res_AR = ress_table.find_adj_relations()
            total_res_relation += len(res_AR)

            # Now map GT adjacency relations to result
            lMappedAR = []
            for ar in gt_AR:
                try:
                    resFromCell = cell_mapping[ar.fromText]
                    resToCell = cell_mapping[ar.toText]
                    # make a mapped adjacency relation
                    lMappedAR.append(AdjRelation(resFromCell, resToCell, ar.direction))
                except Exception:
                    # no mapping is possible
                    pass

            # compare two list of adjacency relation
            correct_dect = 0
            for ar1 in res_AR:
                for ar2 in lMappedAR:
                    if ar1.isEqual(ar2):
                        correct_dect += 1
                        break

            total_correct_relation += correct_dect

        # handle gt_relations in unmatched gt table
        for gtt_remain in gt_remaining:
            total_gt_relation += len(gtt_remain.find_adj_relations())

        # handle gt_relation in unmatched res table
        for res_remain in remaining_tables:
            total_res_relation += len(res_remain.find_adj_relations())

        retVal = ResultStructure(truePos=total_correct_relation, gtTotal=total_gt_relation, resTotal=total_res_relation)
        return retVal


if __name__ == "__main__":
    cur_path = sys.argv[1]
    # print(cur_path)
    eval("-trackB2", cur_path)
