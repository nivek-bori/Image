import random
import numpy as np
import torch.nn.functional as F

def calculate_iou(bbox1, bbox2):
    cx1, cy1, w1, h1 = bbox1
    cx2, cy2, w2, h2 = bbox2
    
    x1_min, y1_min = cx1 - w1/2, cy1 - h1/2
    x1_max, y1_max = cx1 + w1/2, cy1 + h1/2
    x2_min, y2_min = cx2 - w2/2, cy2 - h2/2
    x2_max, y2_max = cx2 + w2/2, cy2 + h2/2
    
    left = max(x1_min, x2_min)
    top = max(y1_min, y2_min)
    right = min(x1_max, x2_max)
    bottom = min(y1_max, y2_max)
    
    intersection = max(0, right - left) * max(0, bottom - top)
    
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_cost_mat(detections, tracks, shape, iou_threshold, age_max_weight):
    cost = np.zeros(shape)
    for i, det in enumerate(detections): # det = yolo detection obj
        for j, tracklet in enumerate(tracks): # tracklet = (pred_bbox, track)
            iou = calculate_iou(det.xywh[0], tracklet[0])

            # iou threshold not met -> inf cost
            if iou < iou_threshold:
                cost[i, j] = 1e6
            # iou threshold met -> cost function
            else:
                cost[i, j] = 1.0 + age_max_weight + 2.0 # iou max + age max + reid cosine difference max
                cost[i, j] -= iou # iou
                cost[i, j] -= age_max_weight * (1 - 1 / (0.1 * (tracklet[1].end - tracklet[1].start) + 1)) # age
                cost[i, j] -= F.cosine_similarity(det.reid, tracklet[1].reid.get_reid(), dim=0) + 1.0 # cos sim domain is -1 to 1, so normalize cost to 0 to 2
                cost[i, j] *= 10
    
    return cost

# detections and tracks are just sorted, values are returned as is
def greedy_match(detections, tracks, iou_threshold=0.5, age_max_weight=0.2):
    if len(detections) == 0:
        return [], [], tracks
    if len(tracks) == 0:
        return [], detections, []
    
    # calculate cost matrix
    shape = (len(detections), len(tracks))
    cost = calculate_cost_mat(detections, tracks, shape, iou_threshold, age_max_weight)
    
    # match detections to tracklets (rows to columns)
    matches = []
    unmatched_det_idx = set(range(len(detections)))
    unmatched_track_idx = set(range(len(tracks)))
    
    while unmatched_det_idx and unmatched_track_idx:
        lowest_cost = 1.0
        best_det_idx = -1
        best_box_idx = -1
        
        for det_i in unmatched_det_idx:
            for box_i in unmatched_track_idx:
                if cost[det_i, box_i] < lowest_cost:
                    lowest_cost = cost[det_i, box_i]
                    best_det_idx = det_i
                    best_box_idx = box_i
        
        if best_det_idx == -1 or best_box_idx == -1:
            break # no more matches
        else:
            matches.append((detections[best_det_idx], tracks[best_box_idx]))
            unmatched_det_idx.remove(best_det_idx)
            unmatched_track_idx.remove(best_box_idx)
    
    unmatched_dets = [detections[i] for i in unmatched_det_idx]
    unmatched_tracks = [tracks[i] for i in unmatched_track_idx]
    
    return matches, unmatched_dets, unmatched_tracks

def hungarian_match(detections, tracks, iou_threshold=0.5, age_max_weight=0.2, log_flag=True):
    if len(detections) == 0:
        return [], [], tracks
    if len(tracks) == 0:
        return [], detections, []
    
    # calculate cost matrix. row = detections, column = tracks
    n = max(len(detections), len(tracks))
    original_cost = calculate_cost_mat(detections, tracks, (n, n), iou_threshold, age_max_weight)
    cost = original_cost.copy()

    if log_flag:
        print('original cost')
        print(cost, '\n')

    # hungarian algorithm step 1 & 2
    cost = cost - cost.min(axis=1, keepdims=True) # limit reduction amount so that 1e6 remains > 1e5
    cost = cost - cost.min(axis=0, keepdims=True)


    covered_cost = cost.copy()
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(n, dtype=bool)

    for _while_iter in range(int(1e3)):
        if log_flag:
            print('normalized cost')
            print(cost, '\n')
        # hungarian algorithm step 3
        num_lines = 0

        row_zeros = (covered_cost == 0).sum(axis=1) # count of zeros along row
        col_zeros = (covered_cost == 0).sum(axis=0) # count of zeros along column
        row_max, col_max = row_zeros.max(), col_zeros.max()

        # while there are still zeros
        while row_max != 0 or col_max != 0:
            num_lines += 1

            # maximum number of zeros in row
            if row_max > col_max:
                idx = row_zeros.argmax() # row to draw line on

                covered_cost[idx, :] = -1 # mark row as covered
                row_covered[idx] = True
            # maximum number of zeros in column
            else:
                idx = col_zeros.argmax() # column to draw line on

                covered_cost[:, idx] = -1 # mark column as covered
                col_covered[idx] = True

            # recalculate for next iteration
            row_zeros = (covered_cost == 0).sum(axis=1) # count of zeros along row
            col_zeros = (covered_cost == 0).sum(axis=0) # count of zeros along column
            row_max, col_max = row_zeros.max(), col_zeros.max()

        if num_lines == n:
            break

        # hungarian algorithm step 4
        uncovered_mask = ~row_covered[:, np.newaxis] & ~col_covered[np.newaxis, :]
        double_covered_mask = row_covered[:, np.newaxis] & col_covered[np.newaxis, :]

        min_val = cost[uncovered_mask].min() # minimum uncovered value
        cost[uncovered_mask] -= min_val # if uncovered -> subtract minimum value
        cost[double_covered_mask] += min_val # if double covered -> add minimum value

    # hungarian algorithm step 5
    matches = []
    matched_det_idx = set()
    matched_track_idx = set()

    assigned_cost = cost.copy()

    # match n detections to tracks
    for _iter in range(n):
        if log_flag:
            print(assigned_cost, '\n')

        row_zeros = (assigned_cost == 0).sum(axis=1)
        col_zeros = (assigned_cost == 0).sum(axis=0)

        row_idx = np.where(assigned_cost == 1)[0]
        col_idx = np.where(assigned_cost == 1)[0]

        # row with one zero
        if len(row_idx) > 0:
            row_idx = row_idx[0] # first row with one zero
            col_idx = np.where(assigned_cost[row_idx, :] == 0)[0][0] # zero in corresponding column
        # column with one zero
        elif len(col_idx) > 0:
            col_idx = col_idx[0] # first col with one zero
            row_idx = np.where(assigned_cost[:, col_idx] == 0)[0][0] # zero in corresponding row
        # no rows or columns with one zero = double zeros
        else:
            row_idx = np.where(row_zeros > 0)[0] # arbitrary row zero

            if len(row_idx) != 0:
                row_idx = row_idx[0]
                col_idx = np.where(assigned_cost[row_idx, :] == 0)[0][0] # corresponding column zero
            else: # edge case
                continue

        # clear row and column
        assigned_cost[row_idx, :] = -1
        assigned_cost[:, col_idx] = -1

        # update matches
        if row_idx < len(detections) and col_idx < len(tracks) and original_cost[row_idx, col_idx] < 1e5: # filter out padding matches and iou's inf
            matches.append((detections[row_idx], tracks[col_idx]))
            matched_det_idx.add(row_idx)
            matched_track_idx.add(col_idx)

    # find unmatched
    unmatched_dets = [detections[i] for i in range(len(detections)) if i not in matched_det_idx]
    unmatched_tracks = [tracks[i] for i in range(len(tracks)) if i not in matched_track_idx]

    if log_flag:
        print(len(matches), len(unmatched_dets), len(unmatched_tracks))
    return matches, unmatched_dets, unmatched_tracks