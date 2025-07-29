import cv2
import time
import torch
import random
import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from util.reid import Reid
from util.logs import timer
from util.gamma import apply_gamma
from util.storage import ResourceLoader
from util.clahe import apply_opencv_clahe
from util.kalman_filter import KalmanFilter
from util.matching import greedy_match, hungarian_match
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

def process_detections(detections, frame, reid_model):
    if detections is None:
        return []
    
    batch_images = torch.stack([get_bbox_image(frame, det) for det in detections])
    batch_features = reid_model(batch_images).detach()  # batch calculate reid

    process_detections = np.zeros(len(detections), dtype=object)

    # apply changes through reference
    for i, det in enumerate(detections):
        det.reid = batch_features[i : i + 1][0]  # extract from batch

        # seperate out the single valued items
        det.cls_v = det.cls[0]
        det.conf_v = det.conf[0].numpy()
        det.xywh_v = det.xywh[0].numpy()
        det.xywhn_v = det.xywhn[0].numpy()
        det.xyxy_v = det.xyxy[0].numpy()
        det.xyxyn_v = det.xyxyn[0].numpy()

        process_detections[i] = det

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

    raise Exception('xywh does not exist')

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
    for frame_i, frame in enumerate(frames, start=video_config.frame_start):
        # frame timing
        frame_start_t = time.perf_counter()

        # preprocessing
        with timer('bytetrack preprocessing gamma', timeout_s=1):
            frame = apply_gamma(frame, gamma=1.1)
        with timer('bytetrack preprocessing clahe', timeout_s=1):
            frame = apply_opencv_clahe(frame, clip_limit=2, grid_shape=(8, 8), image_type='bgr')

        # detection
        with timer('bytetrack yolo detection', timeout_s=2):
            curr_det = process_detections(yolo_model(frame)[0].boxes, frame, reid_model)

        # filter detections based on confidence + add reid
        with timer('bytetrack processing detections + reid', timeout_s=2):
            high_det, low_det = classify_detections(curr_det, high_conf_thres, low_conf_thres)

        # get kalman predictions of prev tracks & match high conf det to preds
        with timer('bytetrack tracks k_filter pred + high conf matching', timeout_s=2):
            tracklets = [(track.k_filter.predict(), track) for track in tracks.values()]
            high_matches, high_unmatched_dets, unmatched_tracks = hungarian_match(high_det, tracklets, log_flag=log_config.log_high_conf_matching)

        # get kalman predictions of prev lost tracks + tracks lost this frame & match low conf det to lost preds for recover
        with timer('bytetrack lost tracks k_filter pred + low conf & unmatched track matching', timeout_s=2):
            lost_tracklets = [(track.k_filter.predict(), track) for track in lost_tracks.values()] + unmatched_tracks
            low_matches, _low_unmatched_dets, low_unmatched_tracks = hungarian_match(low_det, lost_tracklets, log_flag=log_config.log_low_conf_matching)

        # updating states
        with timer('bytetrack updating states (data + k_filter update)', timeout_s=1):
            frame_ids = []
            frame_xywhs = []

            # high conf tracking continues
            for det, tracklet in high_matches:
                track = tracklet[1]

                track.end = frame_i
                track.track_time += 1
                track.reid.step_reid(det.reid, conf=det.conf[0])
                track.k_filter.update(convert_bbox(det))  # update only on trusted info
                tracks[track.id] = track

                frame_ids.append(track.id)
                frame_xywhs.append(track.k_filter.get_xywh())

            # new high conf det -> new tracker
            for det in high_unmatched_dets:
                id += 1  # increment id
                bbox = convert_bbox(det)
                track = Track(id, frame_i, bbox, reid_config)

                track.end = frame_i
                track.track_time = 0
                track.reid.step_reid(det.reid, conf=det.conf[0])
                tracks[id] = track

                frame_ids.append(track.id)
                frame_xywhs.append(track.k_filter.get_xywh())

            # lost tracker or unmatched high tracker recovered -> move to tracks
            for det, tracklet in low_matches:
                id, track = tracklet[1].id, tracklet[1]

                track.end = frame_i
                track.reid.step_reid(det.reid, conf=det.conf[0])
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

            # lost dets not recovered
            for tracklet in low_unmatched_tracks:
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

        # annotate frame
        with timer('bytetrack annotate frame', timeout_s=1):
            annotated_frame = frame.copy()

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

            # detection - white/gray, tracklets - green, lost tracklets - red, predictions - blue
            # annotated_frame = annotate_detections(annotated_frame, (curr_det, (255, 255, 255)))  # all detections
            annotated_frame = annotate_detections(annotated_frame, (high_det, (255, 255, 255)), (low_det, (200, 200, 200)) ) # classified detections
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
            if key in [ord('w'), ord(' ')]:
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
