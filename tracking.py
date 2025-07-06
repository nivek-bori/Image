import cv2
import logging
import numpy as np
from ultralytics import YOLO
from util.kalman_filter import KalmanFilter
from util.matching import greedy_match
from util.video import video_to_frames, get_video_frame_count
from util.rendering import annotate_detections, annotate_tracklets, annotate_predictions, annotate_n_predictions
logging.getLogger('ultralytics').setLevel(logging.ERROR)

auto_play = True
log_flag = False

skip_n_frames = 0
filter_track_time = 0

# Initalize
input_file = 'input/video_1.mp4'
model = YOLO('models/yolo11n.pt')

high_conf_thres = 0.5
low_conf_thres = 0.25
max_lost_time = 200

# self byte track
id_counter = 0

tracks = {}
lost_tracks = {}
removed_tracks = []

frames = video_to_frames(input_file)
frames_len = get_video_frame_count(input_file)

# Functions/Classes
def filter_bboxes(detections):
	high_det = list(filter(lambda x: x.conf >= high_conf_thres, detections))
	low_det = list(filter(lambda x: low_conf_thres <= x.conf < high_conf_thres, detections))

	return high_det, low_det

def convert_bbox(detection):
    if hasattr(detection, 'xywh'):
        return detection.xywh[0].cpu().numpy()
    
    raise Exception('xywh does not exist')

def generate_id():
    global id_counter
    id_counter += 1
    return id_counter

class Track:
    def __init__(self, id, start, bbox):
        self.id = id
        self.start = 0
        self.end = 0
        self.track_time = 0
        self.k_filter = KalmanFilter(bbox)
        
    def __str__(self):
        return f"Track(Id: {self.id}, Frames: {self.start}-{self.end}, Track Time: {self.track_time}, KFilter: {self.k_filter.get_bbox()})"

    def __repr__(self):
        return self.__str__()


# Tracking
def self_byte_track():
	for i, frame in enumerate(frames):   
		if i < skip_n_frames:
			continue

		curr_det = model(frame)[0].boxes # RCW RGB format 
		
		# filter detections based on confidence - numpy
		high_det, low_det = filter_bboxes(curr_det)

		# get kalman predictions of prev tracks & match high conf det to preds
		tracklets = [(track.k_filter.predict(), track) for track in tracks.values()]
		high_matches, high_unmatched_dets, unmatched_tracks = greedy_match(high_det, tracklets)
		
		# get kalman predictions of prev lost tracks + tracks lost this frame & match low conf det to lost preds for recover
		lost_tracklets = [(track.k_filter.predict(), track) for track in lost_tracks.values()] + unmatched_tracks
		low_matches, _low_unmatched_dets, low_unmatched_tracks = greedy_match(low_det, lost_tracklets)
		
		# updating states
		# high conf tracking continues
		for det, tracklet in high_matches:
			track = tracklet[1]

			track.end = i
			track.track_time += 1
			track.k_filter.update(convert_bbox(det)) # update only on trusted info
			tracks[track.id] = track
		
		# new high conf det -> new tracker
		for det in high_unmatched_dets:
			id = generate_id()
			bbox = convert_bbox(det)
			track = Track(id, i, bbox)

			track.end = i
			track.track_time = 0
			tracks[id] = track
		
		# lost tracker or unmatched high tracker recovered -> move to tracks
		for det, tracklet in low_matches:
			id, track = tracklet[1].id, tracklet[1]

			# tracking continues
			if track.track_time >= 0:
				track.end = i
				track.track_time += 1
				track.k_filter.update(convert_bbox(det)) # update only on trusted info
				tracks[id] = track
				
			# move to tracked
			elif track.track_time < 0:
				track.end = i
				track.track_time = 0
				track.k_filter.update(convert_bbox(det)) # update only on trusted info

				if id in lost_tracks:
					del lost_tracks[id]
				tracks[id] = track

		# lost dets not recovered
		for tracklet in low_unmatched_tracks:
			id, track = tracklet[1].id, tracklet[1]

			# track just lost
			if track.track_time >= 0:
				track.end = i
				track.track_time = -1
				
				if id in tracks:
					del tracks[id]
				lost_tracks[id] = track
			
			# track not lost just now
			elif track.track_time < 0:
				track.end = i
				track.track_time -= 1

				# lost time exceeded
				if track.track_time * -1 > max_lost_time:
					removed_tracks.append(lost_tracks[id]) # save it
					if id in lost_tracks:
						del lost_tracks[id] # delete it
				else:
					lost_tracks[id] = track
				
		# discard low unmatched dets because they are probably background
		
		# annotate frame
		annotated_frame = frame.copy()

		filtered_tracks = tracks
		filtered_lost_tracks = lost_tracks
		if filter_track_time > 0:
			filtered_tracks = {id: track for id, track in tracks.items() if track.track_time >= filter_track_time}
			filtered_lost_tracks = {id: track for id, track in lost_tracks.items() if abs(track.track_time) >= filter_track_time}

		# detection - white/gray, tracklets - green, lost tracklets - red, predictions - blue
		# annotated_frame = annotate_detections( annotated_frame, (curr_det, (255, 255, 255)) ) # all detections
		annotated_frame = annotate_detections( annotated_frame, (high_det, (255, 255, 255)), (low_det, (200, 200, 200)) ) # classified detections
		annotated_frame = annotate_tracklets( annotated_frame, filtered_tracks, (0, 255, 0) )
		# annotated_frame = annotate_predictions( annotated_frame, filtered_tracks, (255, 0, 0) )
		# annotated_frame = annotate_n_predictions( annotated_frame, [track.k_filter.predict_n_steps(steps=5, stride=7) for track in filtered_tracks.values()], (255, 0, 0) )
		# annotated_frame = annotate_predictions( annotated_frame, filtered_lost_tracks, (0, 0, 255) )
		# annotated_frame = annotate_n_predictions( annotated_frame, [filtered_lost_tracks.k_filter.predict_n_steps(5, 5) for track in filtered_lost_tracks.values()], (255, 0, 0) )

		# rendering
		cv2.imshow('Object Tracking', annotated_frame)
		if log_flag:
			(f'{i}/{frames_len} -  dets: {len(curr_det)}, f_tracks: {len(filtered_tracks)}, f_lost_tracks: {len(filtered_lost_tracks)}')
		
		if auto_play:
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			key = cv2.waitKey(0) & 0xFF
			if key == ord('q'):
				break

		
	cv2.destroyAllWindows()

def ultra_byte_track():
	model.track(input_file, show=True, save=False, tracker='models/bytetracker.yaml', show_boxes=True, show_labels=False, line_width=2)