import numpy as np


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


def calculate_cost_mat(detections, tracks, shape, iou_threshold, age_max_weight):
    if len(detections) == 0 or len(tracks) == 0:
        return np.zeros(shape)

    # extract important information
    det_xywh = np.array([det.xywh_v for det in detections])[:, np.newaxis]
    det_reid = np.array([det.reid.cpu().numpy() for det in detections])[:, np.newaxis]

    track_xywh = np.array([track[0] for track in tracks])[np.newaxis, :]
    track_time = np.array([track[1].end - track[1].start for track in tracks])[
        np.newaxis, :
    ]
    track_reid = np.array([track[1].reid.get_reid().cpu().numpy() for track in tracks])[
        np.newaxis, :
    ]

    # calculate iou
    iou = calculate_iou(det_xywh, track_xywh)

    # calculate and mask cost
    cost = np.zeros(shape)
    cost[0 : len(detections), 0 : len(tracks)] = np.where(
        iou < iou_threshold,
        1e6,
        10
        * (
            3.0
            + age_max_weight
            - iou
            - age_max_weight * (1 - 1 / (0.1 * track_time + 1))
            - calculate_cosine_similarity(det_reid, track_reid)
            + 1.0
        ),
    )

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
            break  # no more matches
        else:
            matches.append((detections[best_det_idx], tracks[best_box_idx]))
            unmatched_det_idx.remove(best_det_idx)
            unmatched_track_idx.remove(best_box_idx)

    unmatched_dets = [detections[i] for i in unmatched_det_idx]
    unmatched_tracks = [tracks[i] for i in unmatched_track_idx]

    return matches, unmatched_dets, unmatched_tracks


def hungarian_match(
    detections, tracks, iou_threshold=0.5, age_max_weight=0.2, log_flag=True
):
    if len(detections) == 0:
        return [], [], tracks
    if len(tracks) == 0:
        return [], detections, []

    # calculate cost matrix. row = detections, column = tracks
    n = max(len(detections), len(tracks))
    original_cost = calculate_cost_mat(
        detections, tracks, (n, n), iou_threshold, age_max_weight
    )
    cost = original_cost.copy()

    if log_flag:
        print('original cost')
        print(cost, '\n')

    # hungarian algorithm step 1 & 2
    cost = cost - cost.min(
        axis=1, keepdims=True
    )  # limit reduction amount so that 1e6 remains > 1e5
    cost = cost - cost.min(axis=0, keepdims=True)

    # hungarian algorithm step 3 & 4 loop
    covered_cost = cost.copy()
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(n, dtype=bool)

    for _while_iter in range(int(1e3)):
        if log_flag:
            print('normalized cost')
            print(cost, '\n')
        # hungarian algorithm step 3
        num_lines = 0

        row_zeros = (covered_cost == 0).sum(axis=1)  # count of zeros along row
        col_zeros = (covered_cost == 0).sum(axis=0)  # count of zeros along column
        row_max, col_max = row_zeros.max(), col_zeros.max()

        # while there are still zeros
        while row_max != 0 or col_max != 0:
            num_lines += 1

            # maximum number of zeros in row
            if row_max > col_max:
                idx = row_zeros.argmax()  # row to draw line on

                covered_cost[idx, :] = -1  # mark row as covered
                row_covered[idx] = True
            # maximum number of zeros in column
            else:
                idx = col_zeros.argmax()  # column to draw line on

                covered_cost[:, idx] = -1  # mark column as covered
                col_covered[idx] = True

            # recalculate for next iteration
            row_zeros = (covered_cost == 0).sum(axis=1)  # count of zeros along row
            col_zeros = (covered_cost == 0).sum(axis=0)  # count of zeros along column
            row_max, col_max = row_zeros.max(), col_zeros.max()

        if num_lines == n:
            break

        # hungarian algorithm step 4
        uncovered_mask = ~row_covered[:, np.newaxis] & ~col_covered[np.newaxis, :]
        double_covered_mask = row_covered[:, np.newaxis] & col_covered[np.newaxis, :]

        min_val = cost[uncovered_mask].min()  # minimum uncovered value
        cost[uncovered_mask] -= min_val  # if uncovered -> subtract minimum value
        cost[double_covered_mask] += min_val  # if double covered -> add minimum value

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
            row_idx = row_idx[0]  # first row with one zero
            col_idx = np.where(assigned_cost[row_idx, :] == 0)[0][
                0
            ]  # zero in corresponding column
        # column with one zero
        elif len(col_idx) > 0:
            col_idx = col_idx[0]  # first col with one zero
            row_idx = np.where(assigned_cost[:, col_idx] == 0)[0][
                0
            ]  # zero in corresponding row
        # no rows or columns with one zero = double zeros
        else:
            row_idx = np.where(row_zeros > 0)[0]  # arbitrary row zero

            if len(row_idx) != 0:
                row_idx = row_idx[0]
                col_idx = np.where(assigned_cost[row_idx, :] == 0)[0][
                    0
                ]  # corresponding column zero
            else:  # edge case
                continue

        # clear row and column
        assigned_cost[row_idx, :] = -1
        assigned_cost[:, col_idx] = -1

        # update matches
        if (
            row_idx < len(detections)
            and col_idx < len(tracks)
            and original_cost[row_idx, col_idx] < 1e5
        ):  # filter out padding matches and iou's inf
            matches.append((detections[row_idx], tracks[col_idx]))
            matched_det_idx.add(row_idx)
            matched_track_idx.add(col_idx)

    # find unmatched
    unmatched_dets = [
        detections[i] for i in range(len(detections)) if i not in matched_det_idx
    ]
    unmatched_tracks = [
        tracks[i] for i in range(len(tracks)) if i not in matched_track_idx
    ]

    if log_flag:
        print(len(matches), len(unmatched_dets), len(unmatched_tracks))
    return matches, unmatched_dets, unmatched_tracks
