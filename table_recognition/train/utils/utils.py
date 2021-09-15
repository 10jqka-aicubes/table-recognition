import cv2
import numpy as np


# 计算两条线的交点
def calc_line_intersection(line1, line2):
    x1 = float(line1[0][0])
    y1 = float(line1[0][1])
    x2 = float(line1[1][0])
    y2 = float(line1[1][1])
    x3 = float(line2[0][0])
    y3 = float(line2[0][1])
    x4 = float(line2[1][0])
    y4 = float(line2[1][1])

    px = int(
        ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4))
        / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    )
    py = int(
        ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4))
        / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    )
    return [px, py]


def get_polygon_from_lines(lines):
    nb = len(lines)
    polygon = []
    for i in range(nb):
        polygon.append(calc_line_intersection(lines[i], lines[(i + 1) % nb]))

    return polygon


def is_pt_in_polygon(pt, polygon, measureDist=False):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), tuple(pt), measureDist=measureDist)


def convert_pt_to_string(pts):
    pt_str = ""
    for pt in pts:
        if pt_str != "":
            pt_str += " "
        pt_str += str(int(pt[0])) + "," + str(int(pt[1]))

    return pt_str


def convert_string_to_pt(string):
    items = string.split(" ")
    pts = []

    for item in items:
        loc = item.split(",")
        pt = [int(loc[0]), int(loc[1])]
        pts.append(pt)
    return pts


def draw_pts(img, pts):
    length = len(pts)
    for i in range(length):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % length]), (0, 0, 255))


def draw_xml(img, xml_tree):
    root = xml_tree.getroot()
    tables = root.getchildren()

    for table in tables:
        items = table.getchildren()
        for item in items:
            tag = item.tag

            if tag == "Coords":
                pts = convert_string_to_pt(item.attrib["points"])
                draw_pts(img, pts)
            elif tag == "cell":
                cell_loc = item.getchildren()
                pts = convert_string_to_pt(cell_loc[0].attrib["points"])
                draw_pts(img, pts)

    return img
