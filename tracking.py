import cv2
import logging
import numpy as np
from ultralytics import YOLO
from util.video import video_to_frames
from util.kalman_filter import KalmanFilter
from util.matching import greedy_match
from util.rendering import annotate_detections, annotate_tracklets, annotate_predictions, annotate_n_predictions

logging.getLogger('ultralytics').setLevel(logging.ERROR)
auto_play = False

# Initalize
input_file = 'input/video_1.mp4'
model = YOLO('models/yolo11n.pt')

high_conf_thres = 0.5
low_conf_thres = 0.25
max_lost_time = 30

# self byte track
id_counter = 0

tracks = {}
lost_tracks = {}
removed_tracks = []

frames = video_to_frames(input_file)

# Functions/Classes
def filter_bboxes(detections, confidences):
	pairs = list(zip(detections, confidences))
	
	high_det = list(filter(lambda x: x[1] >= high_conf_thres, pairs))
	low_det = list(filter(lambda x: low_conf_thres <= x[1] < high_conf_thres, pairs))
    
	high_det = [det[0] for det in high_det]
	low_det = [det[0] for det in low_det]

	return high_det, low_det

def convert_bbox(detection):
    if hasattr(detection, 'bbox'):
        return detection.bbox
    else:
        return detection  # Already in [x, y, w, h] format

def generate_id():
    global id_counter
    id_counter += 1
    return id_counter

class Track:
    def __init__(self, id, start, bbox):
        self.id = id
        self.start = 0
        self.end = 0
        self.lost_time = 0
        self.k_filter = KalmanFilter(bbox)
        
    def __str__(self):
        return f"Track(Id: {self.id}, Frames: {self.start}-{self.end}, Lost Time: {self.lost_time}, KFilter: {self.k_filter.get_bbox()})"

    def __repr__(self):
        return self.__str__()


# Tracking
def self_byte_track():
	for i, frame in enumerate(frames):    
		curr_det = model(frame)[0].boxes # RCW RGB format
		
		# filter detections based on confidence - numpy
		high_det, low_det = filter_bboxes(curr_det.xywh, curr_det.conf)

		# get kalman predictions of prev tracks & match high conf det to preds
		prev_bboxes = [(id, track.k_filter.predict(), 'tracked') for id, track in tracks.items()]
		high_matches, high_unmatched_dets, unmatched_tracks = greedy_match(high_det, prev_bboxes)
		
		# get kalman predictions of prev lost tracks + tracks lost this frame & match low conf det to lost preds for recover
		prev_lost_bboxes = [(id, track.k_filter.predict(), 'lost') for id, track in lost_tracks.items()] + unmatched_tracks
		low_matches, _low_unmatched_dets, low_unmatched_tracks = greedy_match(low_det, prev_lost_bboxes)
		
		# updating states
		# high conf tracking continues
		for det, box in high_matches:
			id, track = box[0], tracks[box[0]]
			track.end = i
			track.lost_time = 0
			track.k_filter.update(convert_bbox(det)) # update only on trusted info
			tracks[id] = track
		
		# new high conf det -> new tracker
		for det in high_unmatched_dets:
			id = generate_id()
			bbox = convert_bbox(det)
			track = Track(id, i, bbox)

			track.end = i
			track.lost_time = 0
			tracks[id] = track
		
		# lost tracker recovered -> move to tracks
		for det, box in low_matches:
			# tracking continues
			if box[2] == 'tracked':
				id, track = box[0], tracks[box[0]]
				track.end = i
				track.lost_time = 0
				track.k_filter.update(convert_bbox(det)) # update only on trusted info
				tracks[id] = track
				
			# move to tracked
			elif box[2] == 'lost':
				id, track = box[0], lost_tracks[box[0]]
				track.end = i
				track.lost_time = 0
				track.k_filter.update(convert_bbox(det)) # update only on trusted info

				if id in lost_tracks:
					del lost_tracks[id]
				tracks[id] = track

		# lost dets not recovered
		for box in low_unmatched_tracks:
			# track just lost
			if box[2] == 'tracked':
				id, track = box[0], tracks[box[0]]
				track.end = i
				track.lost_time += 1
				
				if id in tracks:
					del tracks[id]
				lost_tracks[id] = track
			
			# track not lost just now
			elif box[2] == 'lost':
				id, track = box[0], lost_tracks[box[0]]
				track.end = i
				track.lost_time += 1

				# lost time exceeded
				if track.lost_time > max_lost_time:
					removed_tracks.append(lost_tracks[id]) # save it
					if id in lost_tracks:
						del lost_tracks[id] # delete it
				else:
					lost_tracks[id] = track
				
		# discard low unmatched dets because they are probably background
		
		# annotate frame
		annotated_frame = frame.copy()
		
		# detection - white, tracklets - green, lost tracklets - red, predictions - blue
		# annotated_frame = annotate_detections(annotated_frame, curr_det, (255, 255, 255))
		annotated_frame = annotate_tracklets(annotated_frame, tracks, (0, 255, 0))
		# annotated_frame = annotate_predictions(annotated_frame, tracks, (255, 0, 0))
		annotated_frame = annotate_n_predictions(annotated_frame, [track.k_filter.predict_n_steps(steps=5, stride=7) for track in tracks.values()], (255, 0, 0))
		# annotated_frame = annotate_predictions(annotated_frame, lost_tracks, (0, 0, 255))
		# annotated_frame = annotate_n_predictions(annotated_frame, [track.k_filter.predict_n_steps(5, 5) for track in lost_tracks.values()], (255, 0, 0))

		# rendering
		# print(f'Frame: {i}, high matches: {len(high_matches)}, high unmatched det: {len(high_unmatched_dets)}, unmatched tracks: {len(unmatched_tracks)}')
		# print(f'Frame: {i}, lost trakcs: {len(lost_tracks)}')
		cv2.imshow('Object Tracking', annotated_frame)
		
		if auto_play:
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			key = cv2.waitKey(0) & 0xFF
			if key == ord('q'):
				break

		
	cv2.destroyAllWindows()

def ultra_byte_track():
	model.track(input_file, show=True, save=False, tracker='models/bytetracker.yaml')

# Main
self_byte_track()
# ultra_byte_track()