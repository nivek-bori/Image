from collections import defaultdict
from sys import orig_argv
import numpy as np
import numpy.ma as ma

def calculate_iou(xywh1, xywh2):
    # calculate intersection
    x1 = np.maximum(xywh1[:, :, 0], xywh2[:, :, 0])
    y1 = np.maximum(xywh1[:, :, 1], xywh2[:, :, 1])
    x2 = np.minimum(xywh1[:, :, 0] + xywh1[:, :, 2], xywh2[:, :, 0] + xywh2[:, :, 2])
    y2 = np.minimum(xywh1[:, :, 1] + xywh1[:, :, 3], xywh2[:, :, 1] + xywh2[:, :, 3])

    width = np.maximum(0, x2 - x1)
    height = np.maximum(0, y2 - y1)
    intersection = width * height

    # calculate area
    area1 = xywh1[:, :, 2] * xywh1[:, :, 3]
    area2 = xywh2[:, :, 2] * xywh2[:, :, 3]
    area = area1 + area2

    # calculate union
    union = area - intersection

    # calculate iou
    iou = np.where(union > 0, intersection / union, 0)
    return iou

def calculate_cosine_similarity(a, b):
    # normalize vectors
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)

    # compute dot product
    return np.sum(a_norm * b_norm, axis=-1)

def calculate_hungarian_cost_mat(detections, tracks, shape=None, iou_threshold=1e5, age_max_weight=1.0):
    if len(detections) == 0 or len(tracks) == 0:
        return np.zeros(shape)
    
    if shape is None:
        shape = (len(detections), len(tracks))

    # extract important information
    det_xywh = np.array([det.xywh_v for det in detections])[:, np.newaxis]
    det_reid = np.array([det.reid.cpu().numpy() for det in detections])[:, np.newaxis]

    track_xywh = np.array([track[0] for track in tracks])[np.newaxis, :]
    track_time = np.array([track[1].end - track[1].start for track in tracks])[np.newaxis, :]
    track_reid = np.array([track[1].reid.get_reid().cpu().numpy() for track in tracks])[np.newaxis, :]

    # calculate iou
    iou = calculate_iou(det_xywh, track_xywh)

    # calculate and mask cost
    cost = np.zeros(shape)
    cost[0 : len(detections), 0 : len(tracks)] = np.where(iou < iou_threshold, 1e6, 10 * ((3.0 + age_max_weight) - iou - age_max_weight * (1 - 1 / (0.1 * track_time + 1)) - (calculate_cosine_similarity(det_reid, track_reid) + 1.0)),)

    return cost

def calculate_minimal_line_cover(rows, cols):
    all_zeros = set(zip(rows, cols))
    all_rows = set(rows)
    all_cols = set(cols)

    # maximally match - augmenting method
    row_graph = defaultdict(list)
    for r, c in all_zeros:
        row_graph[r].append(c)

    row_to_col = {}
    col_to_row = {}

    # True: assignment successful. False: assignment unsuccessful
    def dfs_augmenting_path(row, visited):
        if row in visited:
            return False
        visited.add(row)

        for col in row_graph[row]:
            # if column not matched -> match
            if col not in col_to_row:
                row_to_col[row] = col
                col_to_row[col] = row
                return True

            # if column matched -> reassign the row that the column is matched to
            matched_row = col_to_row[col]
            if dfs_augmenting_path(matched_row, visited):
                row_to_col[row] = col
                col_to_row[col] = row
                return True

        return False

    # for all unmatched rows, match them to a column
    for row in all_rows:
        if row not in row_to_col:
            visited = set() # is per section of matching because rematchings can mean that nodes are revisited
            dfs_augmenting_path(row, visited)

    # minimally cover - ticking method
    marked_rows = all_rows - set(row_to_col.keys()) # initially all unassigned rows
    marked_cols = set() # initially nothingm
    
    changed = True
    while changed:
        changed = False
        
        # mark columns with zeros in marked rows
        for r in marked_rows:
            for c in all_cols:
                if (r, c) in all_zeros and c not in marked_cols:
                    marked_cols.add(c)
                    changed = True
        
        # mark rows with assigned zeros in marked columns
        for c in marked_cols: # marked column
            if c in col_to_row.keys(): # assigned zeros
                matched_row = col_to_row.get(c)
                if matched_row not in marked_rows:
                    marked_rows.add(matched_row)
                    changed = True
    
    line_rows = all_rows - marked_rows # line through all unmarked rows
    line_cols = marked_cols # line through all marked columns

    return list(line_rows), list(line_cols), len(row_to_col)

# detections and tracks are just sorted, values are returned as is
def greedy_match(detections, tracks, iou_threshold=0.5, age_max_weight=0.2):
    if len(detections) == 0:
        return [], [], tracks
    if len(tracks) == 0:
        return [], detections, []

    # calculate cost matrix
    shape = (len(detections), len(tracks))
    cost = calculate_hungarian_cost_mat(detections, tracks, shape, iou_threshold, age_max_weight)

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
            break  # no more matches
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
    original_cost = calculate_hungarian_cost_mat(detections, tracks, (n, n), iou_threshold, age_max_weight)

# add back in for hungarian matching testing
# def temp_hungarian_match(original_cost, log_flag):
#     n = original_cost.shape[0]
#     detections, tracks = range(n), range(n)

    # instead of keeping track of vertice weights, edge weigths are manipulated to represent the excess weight of its nodes. Future references to vertices will be implemented through manipulating node weights
    cost = original_cost.copy()

    if log_flag:
        print(f'step 0\n{cost}\n')

    # hungarian algorithm step 1 & 2
    cost = cost - cost.min(axis=1, keepdims=True) # normalize row (noramlized excess weight of row verticies)
    cost = cost - cost.min(axis=0, keepdims=True) # noramlize columns (noramlized excess weight of column veritices)

    if log_flag:
        print(f'step 1 & 2\n{cost}\n')

    # hungarian algorithm step 3 & 4 loop
    for while_iter_c in range(int(1e2)):
        # hungarian algorithm step 3
        row_zeros, col_zeros = np.where(cost == 0) # the zeros are the equality graph (no excess weight)

        row_lines, col_lines, num_lines = calculate_minimal_line_cover(row_zeros, col_zeros)

        # if optimal matching is possible
        if num_lines == n:
            if log_flag:
                print(f'step 3 - finish iter: {while_iter_c}\n{cost}\n')
            break
        # otherwise create more zeros (reducing excess vertice weights)

        # hungarian algorithm step 4
        line_count = np.zeros_like(cost)
        line_count[row_lines, :] += 1
        line_count[:, col_lines] += 1

        # minimum value of uncovered
        min_val = ma.where(line_count == 0, cost, ma.masked).min()

        if log_flag:
            print(
f'''step 4 - iter: {while_iter_c}, change: {min_val}, num_matches: {num_lines}
lines:
{row_lines} & {col_lines}
cost:
{cost}
lines:
{ma.where(line_count == 0, cost, ma.where(line_count == 2, -1 * cost, ma.masked))}
'''
            )

        cost += np.where(line_count == 0, -min_val, 0) # subtracting mininum value to all uncovered
        cost += np.where(line_count == 2, min_val, 0) # adding minimum value to all doubled covered

    # hungarian algorithm step 5
    matches = []
    unmatched_dets = []
    unmatched_tracks = []

    for while_iter_d in range(int(1e3)):
        all_zeros = np.argwhere(cost == 0)

        if len(all_zeros) == 0:
            break

        row_zeros = (cost == 0).sum(axis=1)
        col_zeros = (cost == 0).sum(axis=0)

        found_match = False

        for r, c in all_zeros:
            if row_zeros[r] == 1 or col_zeros[c] == 1: # required match
                cost[r, :] = -1
                cost[:, c] = -1
                if r < len(detections) and c < len(tracks):
                    if original_cost[r, c] < 1e6:
                        matches.append((detections[r], tracks[c]))
                    else:
                        unmatched_tracks.append(tracks[c])
                        unmatched_dets.append(detections[r])
                elif r >= len(detections) and c < len(tracks):
                    unmatched_tracks.append(tracks[c])
                elif c >= len(tracks) and r < len(detections):
                    unmatched_dets.append(detections[r])
                else:
                    raise Exception('Invalid matching')
                found_match = True
                break
        
        if not found_match:        
            # if no required match -> match arbitrary
            r, c = all_zeros[0]
            cost[r, :] = -1
            cost[:, c] = -1
            if r < len(detections) and c < len(tracks):
                if original_cost[r, c] < 1e6:
                    matches.append((detections[r], tracks[c]))
                else:
                    unmatched_tracks.append(tracks[c])
                    unmatched_dets.append(detections[r])
            elif r >= len(detections) and c < len(tracks):
                unmatched_tracks.append(tracks[c])
            elif c >= len(tracks) and r < len(detections):
                unmatched_dets.append(detections[r])
            else:
                raise Exception('Invalid matching')
        
    if log_flag:
        print(len(matches), len(unmatched_dets), len(unmatched_tracks))
    return matches, unmatched_dets, unmatched_tracks

if __name__ == '__main__':
    mat_type = '8'
    
    match mat_type:
        case '5':
            data_array = [[16, 41, 73, 82, 98], [80, 68, 6, 78, 18], [80, 39, 24, 43, 14], [64, 56, 64, 15, 64], [60, 95, 13, 65, 22]]
            answer = [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
        case '6':
            data_array = [[80, 30, 76, 61, 19, 24], [16, 59, 60, 90, 6, 58], [76, 21, 22, 51, 99, 4], [11, 52, 69, 23, 27, 76], [75, 27, 37, 99, 19, 48], [35, 4, 55, 94, 26, 23]]
            answer = [[0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
        case '7':
            data_array = [[6, 21, 61, 26, 43, 68, 75], [28, 69, 48, 23, 1, 77, 21], [39, 66, 8, 56, 25, 34, 83], [66, 91, 95, 88, 75, 84, 16], [12, 5, 56, 16, 90, 70, 51], [66, 77, 51, 23, 5, 41, 94], [74, 70, 51, 82, 11, 14, 25]]
            answer = [[1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0]]
        case '8':
            data_array = [[16, 10, 41, 15, 14, 68, 3, 67], [95, 75, 32, 38, 44, 68, 70, 85], [29, 67, 57, 25, 71, 31, 20, 63], [7, 92, 65, 91, 34, 25, 19, 99], [67, 64, 66, 69, 40, 42, 26, 71], [7, 3, 19, 31, 1, 82, 88, 37], [69, 4, 35, 50, 21, 5, 69, 18], [71, 15, 21, 64, 20, 29, 48, 35]]
            answer = [[0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]]
    data_array = np.array(data_array)
    answer = np.array(answer)

    # to use this function, slice the hungarian_match function and replace the cost function with the data_array, create temporary detections and tracks
    ret = np.array(temp_hungarian_match(data_array, log_flag=True)[0])
    print('returned matches:\n', ret)

    matches = np.zeros_like(data_array)
    for pos in ret:
        matches[pos[0], pos[1]] = 1

    print('final answer:\n', np.array(matches))
    print('true answe:\n', np.array(answer))
    print(f'passed: {(matches == answer).all()}')
    print(f'final cost: {(data_array * matches).sum()}, true cost: {(data_array * answer).sum()}')
