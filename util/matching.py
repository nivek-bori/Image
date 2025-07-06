# TODO: Implement true hungarian match
import numpy as np

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

# detections and tracks are just sorted and are returned as is
def greedy_match(detections, tracks, iou_threshold=0.1, age_max_weight=0.2):
    if len(detections) == 0:
        return [], [], tracks
    if len(tracks) == 0:
        return [], detections, []
    
    iou = np.zeros((len(detections), len(tracks)))
    cost = np.zeros((len(detections), len(tracks)))
    for i, det in enumerate(detections):
        for j, track in enumerate(tracks):
            iou[i, j] = calculate_iou(det.xywh[0], track[0])

            cost[i, j] = 1.0 + age_max_weight # iou max + age max
            cost[i, j] -= iou[i, j] # iou
            cost[i, j] -= age_max_weight * (1 - 1 / (0.1 * (track[1].end - track[1].start) + 1)) # age
    
    matches = []
    unmatched_det_idx = set(range(len(detections)))
    unmatched_box_idx = set(range(len(tracks)))
    
    while unmatched_det_idx and unmatched_box_idx:
        lowest_cost = 1.0
        best_det_idx = -1
        best_box_idx = -1
        
        for det_i in unmatched_det_idx:
            for box_i in unmatched_box_idx:
                if cost[det_i, box_i] < lowest_cost and iou[det_i, box_i] > iou_threshold:
                    lowest_cost = cost[det_i, box_i]
                    best_det_idx = det_i
                    best_box_idx = box_i
        
        if best_det_idx == -1 or best_box_idx == -1:
            break # no more matches
        else:
            matches.append((detections[best_det_idx], tracks[best_box_idx]))
            unmatched_det_idx.remove(best_det_idx)
            unmatched_box_idx.remove(best_box_idx)
    
    unmatched_dets = [detections[i] for i in unmatched_det_idx]
    unmatched_boxes = [tracks[i] for i in unmatched_box_idx]
    
    return matches, unmatched_dets, unmatched_boxes