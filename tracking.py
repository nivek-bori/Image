import cv2
import time
import torch
import random
import logging
import itertools
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from util.reid import Reid
from util.logs import timer
from util.gamma import apply_gamma
from util.matching import greedy_match, hungarian_match
from util.clahe import apply_opencv_clahe
from util.kalman_filter import KalmanFilter
from util.video import video_to_frames, get_video_frame_count
from util.load_model import get_reid_output_shape, load_yolo_model, load_reid_model, get_reid_model_input_layer
from util.rendering import annotate_detections, annotate_tracklets, annotate_predictions, annotate_n_predictions
logging.getLogger('ultralytics').setLevel(logging.ERROR)

### CONFIGS
# 	should be moved into a config class, this is so bad :(

auto_play = True # auto progress frames
show_bool = True # render image
log_bool = (True, False, False) # print logs for frame, high conf hungarian matching, low conf hungarian matching

required_tracklet_age = 0 # minimum tracklet age before render
frame_start = 1 * (565) # start of frames to process. (0 * x) for unset
frame_end = 1 * (200 + frame_start) # end of frames to process. (0 * x) for unset
 
reid_type, reid_mult = 'ave', 1.5 # reid lookback type. mult is only for 'time' reid
max_reid_lookback = 3 # how many frames reid pulls features from

high_conf_thres = 0.5 # bytetrack high confidence detection threshold
low_conf_thres = 0.3 # bytetrack low confidence detection threshold
max_lost_time = 50 # tracklet  maximum time lost

# Initialize
input_file = 'input/video_1.mp4'

frames, num_frames = video_to_frames(input_file), get_video_frame_count(input_file)
frame_end = min(num_frames, frame_end) if frame_end != 0 else num_frames

model = load_yolo_model('models/yolo11n.pt')
reid_model = load_reid_model('osnet_x0_25')

reid_output_shape = get_reid_output_shape(reid_model)
if isinstance(reid_output_shape, int):
	reid_output_shape = (reid_output_shape, ) # ensure that output_shape is unpackable

### Functions/Classes

def process_bboxes(detections, frame):
    high_det = list(filter(lambda x: x.conf >= high_conf_thres, detections))
    low_det = list(filter(lambda x: low_conf_thres <= x.conf < high_conf_thres, detections))
    
    # batch all detections
    all_dets = high_det + low_det
    if all_dets:
        batch_images = torch.stack([get_bbox_image(frame, det) for det in all_dets])
        batch_features = reid_model(batch_images).detach() # batch calculate reid
        
        for i, det in enumerate(all_dets):
            det.reid = batch_features[i:i+1][0] # extract from batch
    
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
        self.start = start
        self.end = start
        self.track_time = 0
        self.k_filter = KalmanFilter(bbox)
        self.reid = Reid(reid_type, max_reid_lookback, reid_output_shape)
        
    def __str__(self):        
        return f"Track(Id: {self.id}, Frames: {self.start}-{self.end}, Track Time: {self.track_time}, KFilter: {self.k_filter.get_bbox()}), ReID: {self.reid.get_reid()}"

    def __repr__(self):
        return self.__str__()


### Tracking

def self_byte_track():
	id = 0

	tracks = {}
	lost_tracks = {}
	removed_tracks = []

	# slice frame generator to frame_start and frame_end
	for i, frame in enumerate(itertools.islice(frames, frame_start, frame_end), start=frame_start):
		# frame timing
		frame_start_t = time.perf_counter()

		# preprocessing
		with timer('preprocessing gamma', timeout_s=1):
			frame = apply_gamma(frame, gamma=1.1)
		with timer('preprocessing clahe', timeout_s=1):
			frame = apply_opencv_clahe(frame, clip_limit=2, grid_shape=(8, 8), image_type='bgr')

		with timer('yolo', timeout_s=2):
			curr_det = model(frame)[0].boxes # RCW RGB format 
		
		# filter detections based on confidence + add reid
		with timer('processing detections + reid', timeout_s=2):
			high_det, low_det = process_bboxes(curr_det, frame)

		# get kalman predictions of prev tracks & match high conf det to preds
		with timer('tracks k_filter pred + high conf matching', timeout_s=2):
			tracklets = [(track.k_filter.predict(), track) for track in tracks.values()]
			high_matches, high_unmatched_dets, unmatched_tracks = hungarian_match(high_det, tracklets, log_flag=log_bool[1])
			
		# get kalman predictions of prev lost tracks + tracks lost this frame & match low conf det to lost preds for recover
		with timer('lost tracks k_filter pred + low conf & unmatched track matching', timeout_s=2):
			lost_tracklets = [(track.k_filter.predict(), track) for track in lost_tracks.values()] + unmatched_tracks
			low_matches, _low_unmatched_dets, low_unmatched_tracks = hungarian_match(low_det, lost_tracklets, log_flag=log_bool[2])

		# print(len(high_matches), len(high_unmatched_dets), len(unmatched_tracks), len(low_matches), len(_low_unmatched_dets), len(low_unmatched_tracks))
		
		# updating states
		with timer('updating states (data + k_filter update)', timeout_s=1):
			# high conf tracking continues
			for det, tracklet in high_matches:
				track = tracklet[1]

				track.end = i
				track.track_time += 1
				track.reid.step_reid(det.reid, conf=det.conf[0])
				track.k_filter.update(convert_bbox(det)) # update only on trusted info
				tracks[track.id] = track
			
			# new high conf det -> new tracker
			for det in high_unmatched_dets:
				id += 1 # increment id
				bbox = convert_bbox(det)
				track = Track(id, i, bbox)

				track.end = i
				track.track_time = 0
				track.reid.step_reid(det.reid, conf=det.conf[0])
				tracks[id] = track
			
			# lost tracker or unmatched high tracker recovered -> move to tracks
			for det, tracklet in low_matches:
				id, track = tracklet[1].id, tracklet[1]

				track.end = i
				track.reid.step_reid(det.reid, conf=det.conf[0])
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
		with timer('annotate frame', timeout_s=1):
			annotated_frame = frame.copy()

			filtered_tracks = tracks
			filtered_lost_tracks = lost_tracks
			if required_tracklet_age > 0:
				filtered_tracks = {id: track for id, track in tracks.items() if track.track_time >= required_tracklet_age}
				filtered_lost_tracks = {id: track for id, track in lost_tracks.items() if abs(track.track_time) >= required_tracklet_age}

			# detection - white/gray, tracklets - green, lost tracklets - red, predictions - blue
			annotated_frame = annotate_detections( annotated_frame, (curr_det, (255, 255, 255)) ) # all detections
			# annotated_frame = annotate_detections( annotated_frame, (high_det, (255, 255, 255)), (low_det, (200, 200, 200)) ) # classified detections
			annotated_frame = annotate_tracklets( annotated_frame, filtered_tracks, (0, 255, 0) )
			annotated_frame = annotate_tracklets( annotated_frame, filtered_lost_tracks, (0, 0, 255) )
			# annotated_frame = annotate_predictions( annotated_frame, filtered_tracks, (255, 0, 0) )
			# annotated_frame = annotate_n_predictions( annotated_frame, [track.k_filter.predict_n_steps(steps=5, stride=7) for track in filtered_tracks.values()], (255, 0, 0) )
			# annotated_frame = annotate_predictions( annotated_frame, filtered_lost_tracks, (0, 0, 255) )
			# annotated_frame = annotate_n_predictions( annotated_frame, [filtered_lost_tracks.k_filter.predict_n_steps(5, 5) for track in filtered_lost_tracks.values()], (255, 0, 0) )

		frame_end_t = time.perf_counter()
		frame_t = frame_end_t - frame_start_t

		# rendering
		if show_bool:
			cv2.destroyAllWindows()
			cv2.imshow(f'Object Tracking: {i}/{frame_end} {frame_t:.4g}ms', annotated_frame)
		
		if log_bool[0]:
			print(f'{i}/{frame_end} {frame_t:.4g}ms -  dets: {len(curr_det)}, tracks: {len(filtered_tracks)}, lost_tracks: {len(filtered_lost_tracks)}', end='\r')

		if auto_play:
			key = cv2.waitKey(1) & 0xFF # dont wait
			if key == ord('w'):
				cv2.destroyAllWindows()
				return
		else:
			key = cv2.waitKey(0) & 0xFF # do wait
			if key == ord('w'):
				cv2.destroyAllWindows()
				return

	# cleanup
	if log_bool[0]: # log cleanup
		print(100 * ' ', end='\r') # clear line
 
	cv2.destroyAllWindows() # rendering cleanup

def ultra_byte_track():
	model.track(input_file, show=True, save=False, tracker='models/bytetracker.yaml', show_boxes=True, show_labels=False, line_width=2)