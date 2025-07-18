import numpy as np
import motmetrics as mm
from util.config import ByteTrackLogConfig
from util.matching import calculate_iou
from util.rendering import (
    plot_average_metrics,
    plot_camera_comparison,
    plot_error_distribution,
    plot_metric_trends,
)
from util.util import keyboard_quitter


class MOTDataFrame:
    def __init__(self, frame, ids, xywhs):
        self.frame = frame
        self.ids = ids
        self.xywhs = np.array(xywhs)


def calculate_cost_mat(gt_xywh, ts_xywh):
    # format ground truth and tracklet xywh arrays into 2D mat
    row_xywh = gt_xywh[:, np.newaxis, :]
    col_xywh = ts_xywh[np.newaxis, :, :]

    # calculate iou
    iou = calculate_iou(row_xywh, col_xywh)

    # calculate cost
    cost = np.full_like(iou, fill_value=1.0)
    cost += 1 - iou
    return cost


def evaluate_results(gt_list, ts_list):
    if len(gt_list) != len(ts_list):
        raise ValueError(
            'Length of ground truths array and tracklets must be same array'
        )

    acc = mm.MOTAccumulator(auto_id=False)

    for gt, ts in zip(gt_list, ts_list):
        frame = gt.frame

        gt_ids = gt.ids
        ts_ids = ts.ids

        cost_mat = calculate_cost_mat(gt.xywhs, ts.xywhs)

        acc.update(gt_ids, ts_ids, cost_mat, frameid=frame)

    # all_metrics: 'motp', 'mota', 'precision', 'recall', 'id_global_assignment', 'idfp', 'idfn', 'idtp', 'idp', 'idr', 'idf1', 'num_frames', 'obj_frequencies', 'pred_frequencies', 'num_matches', 'num_switches', 'num_transfer', 'num_ascend', 'num_migrate', 'num_false_positives', 'num_misses', 'num_detections', 'num_objects', 'num_predictions', 'num_unique_objects', 'track_ratios', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_fragmentations'
    mh = mm.metrics.create()
    print(mh.__dict__)
    summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'idfp', 'idfn'], name='ByteTrack')

    return summary

# waymo preception evaluation
def evaulate_waymo(file_path, visual_flag=False):
    import tensorflow as tf
    from tracking import self_byte_track
    from waymo_od.waymo_open_dataset import dataset_pb2

    dataset = tf.data.TFRecordDataset(file_path, compression_type='')

    # video from each camera
    camera_frames = {}
    camera_gt = {}

    # for all frames
    for frame_i, frame_data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(frame_data.numpy()))

        # extract per camera image
        for image_data in frame.images:
            camera_name = dataset_pb2.CameraName.Name.Name(image_data.name)
            image = tf.io.decode_jpeg(image_data.image)

            if not camera_name in camera_frames:
                camera_frames[camera_name] = []
            camera_frames[camera_name].append(image)

        # extract ground truth data
        for camera_labels in frame.camera_labels:
            camera_name = dataset_pb2.CameraName.Name.Name(camera_labels.name)

            ids = []
            xywhs = []

            for label in camera_labels.labels:
                # id
                id = label.id

                # cxywh
                x = label.box.center_x - label.box.length / 2
                y = label.box.center_y - label.box.width / 2
                width = label.box.width
                height = label.box.length

                ids.append(id)
                xywhs.append([x, y, width, height])

            gt_result = MOTDataFrame(frame_i, ids, xywhs)

            if not camera_name in camera_gt:
                camera_gt[camera_name] = []
            camera_gt[camera_name].append(gt_result)

    # process frames for tracks
    camera_ts = {}

    for name, frames in camera_name:
        bt_log_config = ByteTrackLogConfig(all_logs=False)
        ts_result = self_byte_track(
            frames, log_config=bt_log_config, render_config=None, reid_config=None
        )  # default render and reid configs

        camera_ts[name] = ts_result

    # combine ground truths and tracklets
    common_names = camera_gt.keys() & camera_ts.keys()
    eval_summaries = {
        name: evaluate_results(camera_gt[name], camera_ts[name])
        for name in common_names
    }

    if visual_flag:
        plot_average_metrics(
            eval_summaries, title='ByteTrack average performance, all cameras'
        )

        plot_camera_comparison(
            eval_summaries, metric_name='mota', title_prefix='ByteTrack performance'
        )

        plot_camera_comparison(
            eval_summaries, metric_name='idf1', title_prefix='ByteTrack performance'
        )

        plot_error_distribution(
            eval_summaries, title='ByteTrack average error distribution'
        )

        plot_metric_trends(
            eval_summaries, metric_name='mota', title_prefix='ByteTrack MOTA trend'
        )

    return eval_summaries

# dummy data
def evaluate_dummy():
    # Create fake ground truth (gt) and tracker (ts) data
    num_frames = 10
    num_ids = 3
    np.random.seed(42)

    # Generate ground truth: each frame has num_ids objects with random positions
    gt_list = []
    for frame in range(num_frames):
        ids = list(range(num_ids))
        xywhs = np.random.rand(num_ids, 4) * 100  # random boxes in [0, 100)
        gt_list.append(MOTDataFrame(frame, ids, xywhs))

    # Generate tracker output: add some noise to gt positions, drop/add some ids
    ts_list = []
    for frame in range(num_frames):
        # Simulate missed detection and false positive
        ts_ids = list(range(num_ids))
        if frame % 3 == 0:
            ts_ids = ts_ids[:-1]  # drop last id
        if frame % 4 == 0:
            ts_ids = ts_ids + [num_ids]  # add a false positive id
        ts_xywhs = np.random.rand(len(ts_ids), 4) * 100  # random boxes
        ts_list.append(MOTDataFrame(frame, ts_ids, ts_xywhs))

    # Evaluate
    summary = evaluate_results(gt_list, ts_list)

    # Visualization
    flat_summary = {k: v['ByteTrack'] if isinstance(v, dict) and 'ByteTrack' in v else v for k, v in summary.to_dict().items()} # Flatten the summary so each value is a number, not a dict
    eval_summaries = {'dummy': flat_summary}
    plot_average_metrics(eval_summaries, title='Dummy average performance')
    plot_camera_comparison(eval_summaries, metric_name='mota', title_prefix='Dummy performance')
    plot_camera_comparison(eval_summaries, metric_name='idf1', title_prefix='Dummy performance')
    plot_error_distribution(eval_summaries, title='Dummy average error distribution')
    plot_metric_trends(eval_summaries, metric_name='mota', title_prefix='Dummy MOTA trend')


# CLI parameters
if __name__ == '__main__':
    import os
    import sys

    args = sys.argv

    # waymo: waymo-open-dataset dependency, only supported on linux/google colab
    if args[1] in ['waymo', 'w']:
        file_path = f'input/waymo_data_{args[2]}.tfrecord'

        # check if file_path exists
        if not os.path.exists(file_path):
            raise ValueError(f'Waymo dataset {args[2]} not found in input/')

        try:
            print('waymo evaluation started')
            keyboard_quitter(evaulate_waymo(file_path, True))
        except Exception as e:
            raise e
        finally:
            print('waymo evaluation ended')
    
    # dummy
    if args[1] in ['dummy', 'd']:
        try:
            print('dummy evaluation started')
            keyboard_quitter(evaluate_dummy)
        except Exception as e:
            raise e
        finally:
            print('dummy evaluation ended')
