from multiprocessing.connection import answer_challenge
import cv2
import time
import torch
import random
import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from util import video
from util.reid import Reid
from util.logs import timer
from util.gamma import apply_gamma
from util.storage import ResourceLoader
from util.clahe import apply_opencv_clahe
from util.kalman_filter import KalmanFilter
from util.matching import greedy_match, hungarian_match, calculate_hungarian_cost_mat
from util.util import MOTDataFrame, slice_generator
from util.video import video_to_frames, get_video_frame_count
from util.config import ByteTrackLogConfig, ReidConfig, ByteTrackVideoConfig
from util.load_model import (
    get_reid_output_shape,
    load_yolo_model,
    load_reid_model,
    get_reid_model_input_layer,
)
from util.rendering import (
    annotate_detections,
    annotate_tracklets,
    annotate_predictions,
    annotate_n_predictions,
)

logging.getLogger('ultralytics').setLevel(logging.ERROR)

### Functions/Classes

class DetectionData:
    def __init__(self, conf, reid, xywh):
        self.conf = conf
        self.reid = reid
        self.xywh_v = xywh

def process_detections(detections, frame, reid_model):
    if detections is None:
        return []
    
    batch_images = torch.stack([get_bbox_image(frame, det) for det in detections])
    batch_features = reid_model(batch_images).detach()  # batch calculate reid

    process_detections = np.zeros(len(detections), dtype=object)

    # apply changes through reference
    for i, det in enumerate(detections):
        if hasattr(det, 'conf'):
            conf = det.conf[0]
        elif 'conf' in det:
            conf = det['conf']
        else:
            raise Exception('detection did not have conf')

        reid = batch_features[i : i + 1][0]  # extract from batch

        if hasattr(det, 'xywh'):
            xywh = det.xywh[0].numpy()
        elif 'xywh' in det:
            xywh = det['xywh']
        else:
            raise Exception('detection did not have xywh')

        # seperate out the single valued items & format
        det_data = DetectionData(
            conf,
            reid,
            xywh,
        )

        process_detections[i] = det_data

    return process_detections     

def classify_detections(detections, high_conf_thres, low_conf_thres):
    high_det = list(filter(lambda x: x.conf >= high_conf_thres, detections))
    low_det = list(filter(lambda x: low_conf_thres <= x.conf < high_conf_thres, detections))

    return high_det, low_det

def convert_bbox(detection):
    if hasattr(detection, 'xywh_v'):
        return detection.xywh_v
    elif hasattr(detection, 'xywh'):
        return detection.xywh[0].cpu().numpy()
    elif 'xywh_v' in detection:
        return detection['xywh_v']
    elif 'xywh' in detection:
        return detection['xywh']

    raise Exception('xywh does not exist')

def matching_factory_a(tracks, lost_tracks, high_det, low_det, log_config):
    # get kalman predictions of prev tracks & match high conf det to preds
    with timer('bytetrack tracks k_filter pred + high conf matching', timeout_s=2):
        tracklets = [(track.k_filter.predict(), track) for track in tracks.values()]
        high_matches, high_unmatched_dets, unmatched_tracks = hungarian_match(high_det, tracklets, log_flag=log_config.log_high_conf_matching)

    # get kalman predictions of prev lost tracks + tracks lost this frame & match low conf det to lost preds for recover
    with timer('bytetrack lost tracks k_filter pred + low conf & unmatched track matching', timeout_s=2):
        lost_tracklets = [(track.k_filter.predict(), track) for track in lost_tracks.values()] + unmatched_tracks
        low_matches, low_unmatched_dets, low_unmatched_tracks = hungarian_match(low_det, lost_tracklets, log_flag=log_config.log_low_conf_matching)

    # continue_tracks, new, recover, continue_lost, unmatched

    continue_tracks = high_matches
    new = high_unmatched_dets
    recover = low_matches
    continue_lost = low_unmatched_tracks
    unmatched = low_unmatched_dets

    return continue_tracks, new, recover, continue_lost, unmatched

def matching_factory_b(tracks, lost_tracks, high_det, low_det, log_config):
    # TODO: finish
    return

transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 128)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_bbox_image(frame, detection):
    # get bbox image
    cx, cy, w, h = convert_bbox(detection)
    x1, y1, x2, y2 = (
        int(cx - w / 2),
        int(cy - h / 2),
        int(cx + w / 2),
        int(cy + h / 2),
    )  # bbox coords
    x1, y1, x2, y2 = (
        max(0, x1),
        max(0, y1),
        min(frame.shape[1], x2),
        min(frame.shape[0], y2),
    )  # bound coords

    image = frame[y1:y2, x1:x2]

    # convert to format that reid model can take
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = transform(image)
    return tensor


class Track:
    def __init__(self, id, start, bbox, reid_config):
        self.id = id
        self.start = start
        self.end = start
        self.track_time = 0
        self.k_filter = KalmanFilter(bbox)
        self.reid = Reid(reid_config)

    def __str__(self):
        return f'Track(Id: {self.id}, Frames: {self.start}-{self.end}, Track Time: {self.track_time}, KFilter: {self.k_filter.get_cxywh()}), ReID: {self.reid.get_reid()}'

    def __repr__(self):
        return self.__str__()


### Tracking

def self_byte_track(input, high_conf_thres=0.5, low_conf_thres=0.3, max_lost_time=50, log_config=None, video_config=None, reid_config=None):
    # config
    if log_config is None:
        log_config = ByteTrackLogConfig()
    if video_config is None:
        video_config = ByteTrackVideoConfig()
    if reid_config is None:
        reid_config = ReidConfig()

    # process input
    # input is file path
    if isinstance(input, str):
        frames = video_to_frames(input)

        num_frames = get_video_frame_count(input)
        max_frames = min(num_frames, video_config.frame_end) if video_config.frame_end > 0 else num_frames
    # input is generator or list
    elif hasattr(input, '__iter__'):
        frames = input

        max_frames = '?'
    # error
    else:
        raise ValueError('input must be file path or generator')
    
    frames = slice_generator(frames, video_config.frame_start, video_config.frame_end)

    # loading models
    with timer('bytetrack loading models'):
        resource_loader = ResourceLoader()

        resource_loader.set_load_function('yolo', lambda: load_yolo_model('models/yolo11n.pt'))
        resource_loader.set_load_function('reid', lambda: load_reid_model('osnet_x0_25'))

        yolo_model = resource_loader.get_resource('yolo')
        reid_model = resource_loader.get_resource('reid')

        reid_output_shape = get_reid_output_shape(reid_model)
        if isinstance(reid_output_shape, int):
            reid_output_shape = (reid_output_shape, )  # ensure that output_shape is unpackable
        reid_config.shape = reid_output_shape

    id = 0
    tracks = {}
    lost_tracks = {}
    results = []

    # slice frame generator to frame_start and frame_end
    for frame_i, frame_data in enumerate(frames, start=video_config.frame_start):
        # frame timing
        frame_start_t = time.perf_counter()

        # detection
        with timer('bytetrack yolo detection', timeout_s=2):
            match video_config.data_format:
                case 'video':
                    # preprocessing
                    with timer('bytetrack preprocessing gamma', timeout_s=1):
                        frame_image = apply_gamma(frame_data, gamma=1.1)
                    with timer('bytetrack preprocessing clahe', timeout_s=1):
                        frame_image = apply_opencv_clahe(frame_data, clip_limit=2, grid_shape=(8, 8), image_type='bgr')

                    curr_det = process_detections(yolo_model(frame_data)[0].boxes, frame_data, reid_model)
                case 'mot20':
                    frame_image, detections = frame_data

                    curr_det = process_detections(detections, frame_image, reid_model)
                case _:
                    raise Exception(f'{video_config.data_format} is not a valid ByteTrackVideoConfig data format')

        # display the matching costs between detections and tracks
        if log_config.log_matching_costs and frame_i % log_config.log_matching_cost_cycle == 0:
            if not video_config.auto_play:
                print('to display matching costs, please turn off auto play')
            else:
                with timer('bytetrack detections to tracklets cost'):
                    cost = calculate_hungarian_cost_mat(detections, tracks, (len(detections), len(tracks)), iou_threshold, age_max_weight)
                    for det_i, det in detections:
                        for track_i, track in tracks:
                            annotated_frame = np.array(frame_image.copy() * 0.5).astype(np.uint8) # dim the image for clearer annotations
    
                            annotated_frame = annotate_detections(annotated_frame, ([det], (255, 0, 0)), ([track], (0, 255, 0)))
                            annotated_frame = cv2.putText(annotated_frame, f'cost: {cost[det_i, track_i]}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2) # display cost between det and track
    
                            # display
                            cv2.destroyAllWindows()
                            cv2.imshow(annotated_frame)
                            
                        # dont wait
                        if log_config.auto_play_matching_costs:
                            key = cv2.waitKey(1) & 0xFF
                            if key in [ord('w'), ord(' ')]:
                                cv2.destroyAllWindows()
                                return
                        # do wait
                        else:
                            key = cv2.waitKey(0) & 0xFF  # do wait
                            if key in [ord('w')]:
                                cv2.destroyAllWindows()
                                return

        # filter detections based on confidence + add reid
        with timer('bytetrack processing detections + reid', timeout_s=2):
            high_det, low_det = classify_detections(curr_det, high_conf_thres, low_conf_thres)

        # match detections to tracks & determine update states
        continue_tracks, new, recover, continue_lost, _unmatched = matching_factory_a()

        # updating states
        with timer('bytetrack updating states (data + k_filter update)', timeout_s=1):
            frame_ids = []
            frame_xywhs = []

            # continue
            for det, tracklet in continue_tracks:
                track = tracklet[1]

                track.end = frame_i
                track.track_time += 1
                track.reid.step_reid(det.reid, conf=det.conf)
                track.k_filter.update(convert_bbox(det))  # update only on trusted info
                tracks[track.id] = track

                frame_ids.append(track.id)
                frame_xywhs.append(track.k_filter.get_xywh())

            # new
            for det in new:
                id += 1  # increment id
                bbox = convert_bbox(det)
                track = Track(id, frame_i, bbox, reid_config)

                track.end = frame_i
                track.track_time = 0
                track.reid.step_reid(det.reid, conf=det.conf)
                tracks[id] = track

                frame_ids.append(track.id)
                frame_xywhs.append(track.k_filter.get_xywh())

            # recover
            for det, tracklet in recover:
                id, track = tracklet[1].id, tracklet[1]

                track.end = frame_i
                track.reid.step_reid(det.reid, conf=det.conf)
                track.k_filter.update(convert_bbox(det))  # update only on trusted info

                # tracking continues
                if track.track_time >= 0:
                    track.track_time += 1
                    tracks[id] = track
                # move to tracked
                elif track.track_time < 0:
                    track.track_time = 0

                    if id in lost_tracks:
                        del lost_tracks[id]
                    tracks[id] = track

                frame_ids.append(track.id)
                frame_xywhs.append(track.k_filter.get_xywh())

            # continue lost
            for tracklet in continue_lost:
                id, track = tracklet[1].id, tracklet[1]

                # track just lost
                if track.track_time >= 0:
                    track.end = frame_i
                    track.track_time = -1

                    if id in tracks:
                        del tracks[id]
                    lost_tracks[id] = track

                # track not lost just now
                elif track.track_time < 0:
                    track.end = frame_i
                    track.track_time -= 1

                    # lost time exceeded
                    if track.track_time * -1 > max_lost_time:
                        if id in lost_tracks:
                            del lost_tracks[id]  # delete it
                    else:
                        lost_tracks[id] = track

            # discard low unmatched dets because they are probably background

        # format tracks
        formatted_track = MOTDataFrame(frame_i, frame_ids, frame_xywhs)
        results.append(formatted_track)

        with timer('bytetrack annotate frame', timeout_s=1):
            annotated_frame = np.array(frame_image.copy() * 0.5).astype(np.uint8) # dim the image for clearer annotations

            # don't render tracks that are too young
            filtered_tracks = tracks
            filtered_lost_tracks = lost_tracks
            if video_config.required_tracklet_age > 0:
                filtered_tracks = {
                    id: track
                    for id, track in tracks.items()
                    if track.track_time >= video_config.required_tracklet_age
                }
                filtered_lost_tracks = {
                    id: track
                    for id, track in lost_tracks.items()
                    if abs(track.track_time) >= video_config.required_tracklet_age
                }

            # writing annotations
            # detection - white/gray, tracklets - green, lost tracklets - red, predictions - blue
            match video_config.data_format:
                case 'video':
                    # annotated_frame = annotate_detections(annotated_frame, (curr_det, (255, 255, 255)))  # all detections
                    annotated_frame = annotate_detections(annotated_frame, (high_det, (255, 255, 255)), (low_det, (220, 220, 200)) ) # classified detections
                case 'mot20':
                    annotated_frame = annotate_detections(annotated_frame, (curr_det, (255, 255, 255)), conf=False)  # all detections
            annotated_frame = annotate_tracklets(annotated_frame, filtered_tracks, (0, 255, 0))
            annotated_frame = annotate_tracklets(annotated_frame, filtered_lost_tracks, (0, 0, 255))
            # annotated_frame = annotate_predictions( annotated_frame, filtered_tracks, (255, 0, 0) )
            # annotated_frame = annotate_n_predictions( annotated_frame, [track.k_filter.predict_n_steps(steps=5, stride=7) for track in filtered_tracks.values()], (255, 0, 0) )
            # annotated_frame = annotate_predictions( annotated_frame, filtered_lost_tracks, (0, 0, 255) )
            # annotated_frame = annotate_n_predictions( annotated_frame, [filtered_lost_tracks.k_filter.predict_n_steps(5, 5) for track in filtered_lost_tracks.values()], (255, 0, 0) )

        frame_end_t = time.perf_counter()
        frame_t = frame_end_t - frame_start_t

        # rendering
        log_config.log_cleanup() # initial cleanup

        if log_config.show_bool:
            cv2.destroyAllWindows()
            cv2.imshow(f'Object Tracking: {frame_i}/{max_frames} {frame_t:.4g}ms', annotated_frame)

        if log_config.log_frame_info:
            end_token = '\r' if log_config.temporary_frame_info else '\n'
            print(f'{frame_i}/{max_frames} {frame_t:.4g}ms -  dets: {len(curr_det)}, tracks: {len(filtered_tracks)}, lost_tracks: {len(filtered_lost_tracks)}', end=end_token)

        if log_config.log_results_info:
            end_token = '\r' if log_config.temporary_frame_info else '\n'
            print(f'{frame_i} results: {formatted_track}', end=end_token)

        # dont wait
        if log_config.auto_play:
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('w'), ord(' ')]:
                cv2.destroyAllWindows()
                return
        # do wait
        else:
            key = cv2.waitKey(0) & 0xFF  # do wait
            if key in [ord('w')]:
                cv2.destroyAllWindows()
                return
            
    # final cleanup
    log_config.log_cleanup()

    # return results for evaulation
    return results


def ultra_byte_track(input_file, show_bool=True):
    # process input
    if not isinstance(input, str):
        raise ValueError('input must be file path')

    # loading model
    resource_loader = ResourceLoader()
    resource_loader.set_load_function(
        'yolo', lambda x: load_yolo_model('models/yolo11n.pt')
    )
    yolo_model = resource_loader.get_resource('yolo')

    ultralytics_results = yolo_model.track(
        input_file,
        show=show_bool,
        save=False,
        tracker='models/bytetracker.yaml',
        show_boxes=True,
        show_labels=False,
        line_width=2,
    )

    results = []
    for frame_i, r in enumerate(ultralytics_results):
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id
            xywhs = r.boxes.xywh
            frame_results = MOTDataFrame(frame_i, ids, xywhs)
        else:
            frame_results = MOTDataFrame(frame_i, [], [])

        results.append(frame_results)

    return results
