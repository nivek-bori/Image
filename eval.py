from threading import Thread
import time
from types import SimpleNamespace
import numpy as np
import motmetrics as mm
import matplotlib.pyplot as plt
from util.config import ByteTrackLogConfig, ByteTrackVideoConfig, Mot20Config
from util.rendering import annotate_detections, bar_chart, pie_chart
from util.util import MOTDataFrame, calculate_cost_mat, keyboard_quitter, wait_key_press
from tracking import self_byte_track
from util.logs import Logger, timer

# TODO: FINISH & TEST
def evaluate_hot_results():
    import trackeval

    # Initialize evaluator
    evaluator = trackeval.Evaluator()

    # Set up dataset and metrics
    dataset = trackeval.datasets.MotChallenge2DBox(...)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR()]

    # Run evaluation
    evaluator.evaluate(dataset, metrics)

def evaluate_mot_results(gt_list, ts_list):
    if len(gt_list) != len(ts_list):
        raise ValueError('Length of ground truths array and tracklets must be same array')

    acc = mm.MOTAccumulator(auto_id=False)

    for gt, ts in zip(gt_list, ts_list):
        frame = gt.frame

        gt_ids = gt.ids
        ts_ids = ts.ids

        cost_mat = calculate_cost_mat(gt.xywhs, ts.xywhs)

        acc.update(gt_ids, ts_ids, cost_mat, frameid=frame)

    mh = mm.metrics.create()
    all_metrics = ['motp', 'mota', 'precision', 'recall', 'id_global_assignment', 'idfp', 'idfn', 'idtp', 'idp', 'idr', 'idf1', 'num_frames', 'obj_frequencies', 'pred_frequencies', 'num_matches', 'num_switches', 'num_transfer', 'num_ascend', 'num_migrate', 'num_false_positives', 'num_misses', 'num_detections', 'num_objects', 'num_predictions', 'num_unique_objects', 'track_ratios', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_fragmentations']
    summary = mh.compute(acc, metrics=all_metrics, name='ByteTrack')

    return summary

# mot20 dataset evaluation
def evaulate_mot20(mot_folder_path, mot20_config=None, bytetrack_log_config=None, bytetrack_video_config=None):
    import cv2

    # default configs
    if mot20_config is None:
        mot20_config = Mot20Config()
    if bytetrack_log_config is None:
        bytetrack_log_config = ByteTrackLogConfig()
    if bytetrack_video_config is None:
        bytetrack_video_config = ByteTrackVideoConfig()

    # evaluation data
    gt_list = []
    ts_list = []
    result = None
    
    # ground truth
    with timer('mot20 ground truth'):        
        # read gt data
        data_gt = np.loadtxt(os.path.join(mot_folder_path, 'gt/gt.txt'), delimiter=',', dtype=float)

        frame_gt = {}
        for tracklet in data_gt:
            # read ground truth data
            frame_id = tracklet[0].astype(int) - 1 # frames start on one for mot20
            id = tracklet[1].astype(int)
            xywh = [tracklet[2], tracklet[3], tracklet[4], tracklet[5]]

            # if in valid frame range -> save data
            if bytetrack_video_config.frame_start <= frame_id and frame_id < bytetrack_video_config.frame_end:
                if not frame_id in frame_gt:
                    frame_gt[frame_id] = {'ids': [], 'xywhs': []}
                frame_gt[frame_id]['ids'].append(id)
                frame_gt[frame_id]['xywhs'].append(xywh)

        for frame_id, frame_data in frame_gt.items():
            gt = MOTDataFrame(frame_id, frame_data['ids'], frame_data['xywhs'])
            gt_list.append(gt)

    # tracklets
    with timer('mot20 tracklets'):
        match bytetrack_video_config.data_format:
            case 'video':
                all_files = os.listdir(os.path.join(mot_folder_path, 'img1'))
                images = sorted([f for f in all_files])

                def open_images(images): # a generator
                    for filename in images:
                        filepath = os.path.join(os.path.join(mot_folder_path, 'img1'), filename)
                        img = cv2.imread(filepath)
                        
                        if img is not None:
                            yield np.array(img)
                        else:
                            print(f'Could not load {filename}')
                
                frames_data = open_images(images)
            case 'mot20':
                # read detection data
                data_ts = np.loadtxt(os.path.join(mot_folder_path, 'det/det.txt'), delimiter=',', dtype=float)
                
                frame_ts = {}
                for tracklet in data_ts:
                    # read ground truth data
                    frame_id = tracklet[0].astype(int) - 1 # frames start on one for mot20
                    id = tracklet[1].astype(int)
                    xywh = [tracklet[2] + tracklet[4] / 2, tracklet[3] + tracklet[5] / 2, tracklet[4], tracklet[5]]
                    conf = 1.0

                    if not frame_id in frame_ts:
                        frame_ts[frame_id] = []
                    frame_ts[frame_id].append({'conf': conf, 'xywh': xywh})

                all_files = os.listdir(os.path.join(mot_folder_path, 'img1'))
                images = sorted([f for f in all_files])

                def open_images(images): # a generator
                    for i, filename in enumerate(images):
                        filepath = os.path.join(os.path.join(mot_folder_path, 'img1'), filename)
                        img = cv2.imread(filepath)
                        
                        if img is not None:
                            yield (np.array(img), frame_ts[i])
                        else:
                            print(f'Could not load {filename}')
                
                frames_data = open_images(images)

                filepath = os.path.join(os.path.join(mot_folder_path, 'img1'), images[0])
                img = cv2.imread(filepath)
                frame_shape = img.shape

        
        ts_list = self_byte_track(input=frames_data, log_config=bytetrack_log_config, video_config=bytetrack_video_config)

    # evaluate
    with timer('mot20 evaluation'):
        result = evaluate_mot_results(gt_list, ts_list)
        result = {k: v['ByteTrack'] if isinstance(v, dict) and 'ByteTrack' in v else v for k, v in result.to_dict().items()}  # extrack ByteTrack row and reconstruct original structure

    if mot20_config.render:
        wait_key_press()

        max_frames = len(gt_list)
        for gt, ts in zip(gt_list, ts_list): # for each frame
            frame_start = time.perf_counter()

            # format for annotate_detections
            gt_dets = [SimpleNamespace(xywh_v=xywh) for xywh in gt.xywhs] # SimpleNamespace turns maps into objects
            ts_dets = [SimpleNamespace(xywh_v=xywh) for xywh in ts.xywhs]

            # render
            frame = np.zeros(frame_shape)
            annotated_frame = annotate_detections(frame, (gt_dets, (0, 0, 255)), (ts_dets, (0, 255, 0)), conf=False)
            cv2.imshow(f'Object Tracking: {frame}/{max_frames}', annotated_frame)

            # display
            if mot20_config.auto_play: # dont wait
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('w'), ord(' ')]:
                    cv2.destroyAllWindows()
                    return

                frame_end = time.perf_counter()
                time_sleep = (1/20) - (frame_end - frame_start)
                if time_sleep > 0:
                    time.sleep(time_sleep) # ensure minimum frame rate
            # do wait
            else:
                key = cv2.waitKey(0) & 0xFF  # do wait
                if key in [ord('w')]:
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows() # clean up
        wait_key_press() # ensure user is ready

    if mot20_config.plot_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
        axes = axes.flatten()

        performance_labels_a = ['mota', 'motp (distance based)', 'idf1'] # displayed label name
        performance_data_a = [result[label] for label in ['mota', 'motp', 'idf1']] # metric's label name
        bar_chart(title='Performance Metrics', labels=performance_labels_a, data=performance_data_a, fig=fig, ax=axes[0], show_plot=False)

        performance_labels_b = ['precision', 'recall']
        performance_data_b = [result[label] for label in ['precision', 'recall']]
        bar_chart(title='Performance Metrics', labels=performance_labels_b, data=performance_data_b, fig=fig, ax=axes[1], show_plot=False)

        id_labels = ['idfp', 'idfn', 'idtp']
        id_data = [result[label] for label in ['idfp', 'idfn', 'idtp']]
        pie_chart(title='Identification Metrics', labels=id_labels, data=id_data, fig=fig, ax=axes[2], show_plot=False)

        plt.show()

        all_tracked_metrics = ['mota', 'motp', 'idf1', 'precision', 'recall', 'idfp', 'idfn', 'idtp']
        table = np.array([all_tracked_metrics, [result[label] for label in all_tracked_metrics]])
        print('evaluation metrics:\n', table)

# waymo preception dataset evaluation
def evaulate_waymo(file_path, visual_flag=False):
    import tensorflow as tf
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
            frames, log_config=bt_log_config, video_config=None, reid_config=None
        )  # default render and reid configs

        camera_ts[name] = ts_result

    # combine ground truths and tracklets
    common_names = camera_gt.keys() & camera_ts.keys()
    results = {}
    for name in common_names:
        result = evaluate_mot_results(camera_gt[name], camera_ts[name])
        result = {k: v['ByteTrack'] if isinstance(v, dict) and 'ByteTrack' in v else v for k, v in results.to_dict().items()}  # extrack ByteTrack row and reconstruct original structure
        results[name] = result

    if visual_flag:
        for name, result in results:
            # todo - visualize
            lambda x: x

# dummy data
def evaluate_dummy():
    # dummy configs
    num_frames = 10
    num_ids = 3
    np.random.seed(42)

    gt_list = []
    for frame in range(num_frames):
        ids = list(range(num_ids))
        xywhs = np.random.rand(num_ids, 4) * 100
        gt_list.append(MOTDataFrame(frame, ids, xywhs))

    ts_list = []
    for frame in range(num_frames):
        ts_ids = list(range(num_ids))
        if frame % 3 == 0:
            ts_ids = ts_ids[:-1]  # drop last id
        if frame % 4 == 0:
            ts_ids = ts_ids + [num_ids]  # add a false positive id
        ts_xywhs = np.random.rand(len(ts_ids), 4) * 100  # random boxes
        ts_list.append(MOTDataFrame(frame, ts_ids, ts_xywhs))

    result = evaluate_mot_results(gt_list, ts_list)
    result = {k: v['ByteTrack'] if isinstance(v, dict) and 'ByteTrack' in v else v for k, v in result.to_dict().items()}  # extrack ByteTrack row and reconstruct original structure

    # todo - visualize

# CLI parameters
if __name__ == '__main__':
    import os
    import sys

    args = sys.argv

    # mot20q
    if args[1] in ['mot20', 'mot', 'm']:
        try:
            print('motw20 evaluation started')
            # default configs
            mot20_config = Mot20Config(render=True, auto_play=True)
            bytetrack_log_config = ByteTrackLogConfig(auto_play=False, show_bool=True, log_frame_info=True, log_results_info=False, temporary_frame_info=True)
            bytetrack_video_config = ByteTrackVideoConfig(data_format='mot20', frame_start=0, frame_end=100, frame_shape=(1080, 1920))

            # without keyboard quitter
            if len(args) > 2 and args[2] == '-k':
                results = evaulate_mot20(mot_folder_path='input/MOT20/train/MOT20-01', mot20_config=mot20_config, bytetrack_log_config=bytetrack_log_config, bytetrack_video_config=bytetrack_video_config)
            # with keyboard quitter
            else:
                results = keyboard_quitter(evaulate_mot20, mot_folder_path='input/MOT20/train/MOT20-01', mot20_config=mot20_config, bytetrack_log_config=bytetrack_log_config, bytetrack_video_config=bytetrack_video_config)
        except Exception as e:
            raise e
        finally:
            print('mot20 evaluation ended')

            # logger
            logger = Logger()
            logger.log_timing()

    # waymo: waymo-open-dataset dependency, only supported on linux/google colab
    if args[1] in ['waymo', 'w']:
        try:
            print('waymo evaluation started')
            keyboard_quitter(evaulate_waymo, file_path=f'input/waymo_data_{args[2]}.tfrecord', visual_flag=True)
        except Exception as e:
            raise e
        finally:
            print('waymo evaluation ended')

            logger = Logger()
            logger.log_timing()
    
    # dummy
    if args[1] in ['dummy', 'd']:
        try:
            print('dummy evaluation started')
            keyboard_quitter(evaluate_dummy)
        except Exception as e:
            raise e
        finally:
            print('dummy evaluation ended')

            logger = Logger()
            logger.log_timing()
