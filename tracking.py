import cv2
import torch
import random
import logging
import numpy as np
import torchvision.transforms as transforms
from util.logs import timer
from util.gamma import apply_gamma
from util.matching import greedy_match
from util.clahe import apply_opencv_clahe
from util.kalman_filter import KalmanFilter
from util.video import video_to_frames, get_video_frame_count
from util.load_model import load_yolo_model, load_reid_model, get_reid_model_input_layer
from util.rendering import annotate_detections, annotate_tracklets, annotate_predictions, annotate_n_predictions
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('torchreid').setLevel(logging.ERROR)


### CONFIGS
# 	should be moved into a config class

auto_play = True
show_bool = True
log_bool = True

filter_track_time = 0
skip_n_frames = 0
max_frames = -1 # -1 for no max

high_conf_thres = 0.5
low_conf_thres = 0.25
max_lost_time = 200

# initialize
input_file = 'input/video_1.mp4'
model = load_yolo_model('models/yolo11n.pt')
reid_model = load_reid_model('osnet_x0_25')

frames = video_to_frames(input_file)
frames_len = get_video_frame_count(input_file) if max_frames == -1 else min(max_frames, get_video_frame_count(input_file))


### Functions/Classes

def process_bboxes(detections, frame):
    high_det = list(filter(lambda x: x.conf >= high_conf_thres, detections))
    low_det = list(filter(lambda x: low_conf_thres <= x.conf < high_conf_thres, detections))
    
    # batch all detections
    all_dets = high_det + low_det
    if all_dets:
        batch_images = torch.stack([get_bbox_image(frame, det) for det in all_dets])
        batch_features = reid_model(batch_images) # batch calculate reid
        
        for i, det in enumerate(all_dets):
            det.reid = batch_features[i:i+1] # extract from batch
    
    return high_det, low_det

def convert_bbox(detection):
    if hasattr(detection, 'xywh'):
        return detection.xywh[0].cpu().numpy()
    
    raise Exception('xywh does not exist')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_bbox_image(frame, detection):
	# get bbox image
    cx, cy, w, h = convert_bbox(detection)
    x1, y1, x2, y2 = int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2) # bbox coords
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2) # bound coords
    
    image = frame[y1:y2, x1:x2]

	# convert to format that reid model can take
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    tensor = transform(image)
    return tensor

class Track:
    def __init__(self, id, start, bbox):
        self.id = id
        self.start = 0
        self.end = 0
        self.track_time = 0
        self.k_filter = KalmanFilter(bbox)
        self.reid = None
        
    def __str__(self):        
        return f"Track(Id: {self.id}, Frames: {self.start}-{self.end}, Track Time: {self.track_time}, KFilter: {self.k_filter.get_bbox()}), ReID: {self.reid}"

    def __repr__(self):
        return self.__str__()


### Tracking

def self_byte_track():
	id_counter = 0

	tracks = {}
	lost_tracks = {}
	removed_tracks = []

	for i, frame in enumerate(frames):
		if max_frames >= 0 and i > max_frames:
			break
		if i < skip_n_frames:
			continue

		# preprocessing
		with timer('preprocessing gamma'):
			frame = apply_gamma(frame, gamma=1.1)
		with timer('preprocessing clahe'):
			frame = apply_opencv_clahe(frame, clip_limit=2, num_grids=(8, 8), image_type='bgr')

		with timer('yolo'):
			curr_det = model(frame)[0].boxes # RCW RGB format 
		
		# filter detections based on confidence + add reid
		with timer('processing detections + reid'):
			high_det, low_det = process_bboxes(curr_det, frame)

		# get kalman predictions of prev tracks & match high conf det to preds
		with timer('tracks k_filter pred + high conf matching'):
			tracklets = [(track.k_filter.predict(), track) for track in tracks.values()]
			high_matches, high_unmatched_dets, unmatched_tracks = greedy_match(high_det, tracklets)
			
		# get kalman predictions of prev lost tracks + tracks lost this frame & match low conf det to lost preds for recover
		with timer('lost tracks k_filter pred + low conf & unmatched track matching'):
			lost_tracklets = [(track.k_filter.predict(), track) for track in lost_tracks.values()] + unmatched_tracks
			low_matches, _low_unmatched_dets, low_unmatched_tracks = greedy_match(low_det, lost_tracklets)
		
		# updating states
		with timer('updating states (data + k_filter update)'):
			# high conf tracking continues
			for det, tracklet in high_matches:
				track = tracklet[1]

				track.end = i
				track.track_time += 1
				track.reid = det.reid
				track.k_filter.update(convert_bbox(det)) # update only on trusted info
				tracks[track.id] = track
			
			# new high conf det -> new tracker
			for det in high_unmatched_dets:
				id += 1 # increment id
				bbox = convert_bbox(det)
				track = Track(id, i, bbox)

				track.end = i
				track.track_time = 0
				track.reid = det.reid
				tracks[id] = track
			
			# lost tracker or unmatched high tracker recovered -> move to tracks
			for det, tracklet in low_matches:
				id, track = tracklet[1].id, tracklet[1]

				track.end = i
				track.reid = det.reid
				track.k_filter.update(convert_bbox(det)) # update only on trusted info
				
				# tracking continues 
				if track.track_time >= 0:
					track.track_time += 1
					
				# move to tracked
				elif track.track_time < 0:
					track.track_time = 0

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
		with timer('annotate frame'):
			annotated_frame = frame.copy()

			filtered_tracks = tracks
			filtered_lost_tracks = lost_tracks
			if filter_track_time > 0:
				filtered_tracks = {id: track for id, track in tracks.items() if track.track_time >= filter_track_time}
				filtered_lost_tracks = {id: track for id, track in lost_tracks.items() if abs(track.track_time) >= filter_track_time}

			# detection - white/gray, tracklets - green, lost tracklets - red, predictions - blue
			# annotated_frame = annotate_detections( annotated_frame, (curr_det, (255, 255, 255)) ) # all detections
			# annotated_frame = annotate_detections( annotated_frame, (high_det, (255, 255, 255)), (low_det, (200, 200, 200)) ) # classified detections
			annotated_frame = annotate_tracklets( annotated_frame, filtered_tracks, (0, 255, 0) )
			# annotated_frame = annotate_predictions( annotated_frame, filtered_tracks, (255, 0, 0) )
			annotated_frame = annotate_n_predictions( annotated_frame, [track.k_filter.predict_n_steps(steps=5, stride=7) for track in filtered_tracks.values()], (255, 0, 0) )
			# annotated_frame = annotate_predictions( annotated_frame, filtered_lost_tracks, (0, 0, 255) )
			# annotated_frame = annotate_n_predictions( annotated_frame, [filtered_lost_tracks.k_filter.predict_n_steps(5, 5) for track in filtered_lost_tracks.values()], (255, 0, 0) )

		# rendering
		if show_bool:
			cv2.imshow('Object Tracking', annotated_frame)
		
		if log_bool:
			print(f'{i}/{frames_len} -  dets: {len(curr_det)}, f_tracks: {len(filtered_tracks)}, f_lost_tracks: {len(filtered_lost_tracks)}', end='\r')

		if auto_play:
			if cv2.waitKey(1) & 0xFF == ord('q'): # dont wait
				break
		else:
			key = cv2.waitKey(0) & 0xFF # do wait
			if key == ord('q'):
				break

	if log_bool:
		print(100 * ' ', end='\r') # clear line
	cv2.destroyAllWindows()

def ultra_byte_track():
	model.track(input_file, show=True, save=False, tracker='models/bytetracker.yaml', show_boxes=True, show_labels=False, line_width=2)