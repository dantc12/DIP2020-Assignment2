import copy
from operator import itemgetter
import cv2
import numpy as np


def neighbor_val(arr, x, y):
    try:
        if arr[x-1, y] > 0:
            return 1
    except IndexError:
        pass
    try:
        if arr[x, y-1] > 0:
            return 1
    except IndexError:
        pass
    try:
        if arr[x, y+1] > 0:
            return 1
    except IndexError:
        pass
    try:
        if arr[x+1, y] > 0:
            return 1
    except IndexError:
        pass
    return 0


def make_struct_elem(contour):
    """
    :type contour: numpy.ndarray
    """
    flattened_contour = [c[0] for c in contour.tolist()]
    min_p_y = min(flattened_contour, key=itemgetter(0))[0]
    min_p_x = min(flattened_contour, key=itemgetter(1))[1]
    max_p_y = max(flattened_contour, key=itemgetter(0))[0]
    max_p_x = max(flattened_contour, key=itemgetter(1))[1]
    cnt_normalized = [[p[1]-min_p_x, p[0]-min_p_y] for p in flattened_contour]
    rows_count = max_p_x - min_p_x + 1
    cols_count = max_p_y - min_p_y + 1
    struct_size = rows_count * cols_count
    res = np.zeros(struct_size, dtype=np.uint8).reshape(rows_count, cols_count)
    for (x, y) in cnt_normalized:
        res[x][y] = 1
    for x in range(rows_count):
        inside = False
        inside_start_ind = 0
        for y in range(cols_count):
            if res[x][y] == 1 and not inside:
                inside = True
                inside_start_ind = y
            elif res[x][y] == 1 and inside:
                inside_end_ind = y
                inside = False
                for j in range(inside_start_ind, inside_end_ind):
                    res[x][j] = 1

    fin_res = np.pad(res, pad_width=1, mode='constant', constant_values=0)
    worker_res = copy.copy(fin_res)
    for x in range(rows_count+2):
        for y in range(cols_count+2):
            if worker_res[x][y] == 0:
                if neighbor_val(worker_res, x, y) > 0:
                    fin_res[x][y] = 1
                    return fin_res


def is_cnt_relevant(img, contour):
    flattened_contour = [c[0] for c in contour.tolist()]
    for p in flattened_contour:
        if img[p[1], p[0]] == 255:
            return True
    return False


def remove_demarcation(cnt, curr_img):
    st = make_struct_elem(cnt)
    ksize = st.shape[0]
    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opening_img = cv2.morphologyEx(curr_img, cv2.MORPH_OPEN, struct_elem)

    return opening_img


def rebuild_org_img(org_img, contours, ref_img):
    res_img = np.ones(org_img.shape[:2], dtype=np.uint8) * 255
    for cnt in contours:
        if is_cnt_relevant(ref_img, cnt):
            cv2.drawContours(res_img, [cnt], -1, 0, -1)
    return res_img
