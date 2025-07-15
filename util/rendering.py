import cv2

def annotate_detections(annotated_frame, *boxes):
	# curr_det: [cx, cy, w, h]

	for detections, color in boxes:
		for det in detections:
			xywh = det.xywh[0]
			
			# detection pos
			x1 = int(xywh[0] - xywh[2] / 2)
			x2 = int(xywh[0] + xywh[2] / 2)
			y1 = int(xywh[1] - xywh[3] / 2)
			y2 = int(xywh[1] + xywh[3] / 2)
			
			# detection bounding box
			cv2.circle(annotated_frame, (int(xywh[0]), int(xywh[1])), 6, color, 2) # raw xy
			cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
			
			# detection text
			text = f'conf: {det.conf[0]}'
			cv2.putText(annotated_frame, text, (int(x1), int(y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
	
	return annotated_frame


def annotate_tracklets(annotated_frame, tracks, color):
	for id, track in tracks.items():
		bbox = track.k_filter.x[:, 0] # x: [cx, cy, width, height, vx, vy, vw, vh]
        
        # tracking pos
		w, h = bbox[2], bbox[3]
		x1 = int(bbox[0] - w/2)
		y1 = int(bbox[1] - h/2)
		x2 = int(bbox[0] + w/2)
		y2 = int(bbox[1] + h/2)
        
        # tracking bounding box
		cv2.circle(annotated_frame, (int(bbox[0]), int(bbox[1])), 6, color, 2) # raw cxy
		cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
		# tracking text
		text = f'id: {id}'
		cv2.putText(annotated_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
	
	return annotated_frame
        

def annotate_predictions(annotated_frame, tracks, color):
	for track in tracks.values():
		bbox = track.k_filter.x[:, 0] # x: [cx, cy, width, height, vx, vy, vw, vh]

		# prediction pos
		pred_w, pred_h = bbox[2] + bbox[6], bbox[3] + bbox[7]
		pred_x1 = int( (bbox[0] + bbox[4]) - pred_w/2 )
		pred_y1 = int( (bbox[1] + bbox[5]) - pred_h/2 )
		pred_x2 = int( (bbox[0] + bbox[4]) + pred_w/2 )
		pred_y2 = int( (bbox[1] + bbox[5]) + pred_h/2 )
        
        # prediction bounding box
		cv2.circle(annotated_frame, (int(bbox[0]), int(bbox[1])), 6, color, 2) # raw cxy
		cv2.rectangle(annotated_frame, (pred_x1, pred_y1), (pred_x2, pred_y2), color, 2)
	
	return annotated_frame

def annotate_n_predictions(annotated_frame, pred_tracks, color):
	for track in pred_tracks:
		for i, bbox in enumerate(track[1:]): # do not include first bbox (non-prediction bbox)
			# prediction pos
			w, h = bbox[2, 0], bbox[3, 0]
			x1 = int(bbox[0, 0] - w/2)
			y1 = int(bbox[1, 0] - h/2)
			x2 = int(bbox[0, 0] + w/2)
			y2 = int(bbox[1, 0] + h/2)
			
			# prediction bounding box
			weight = 1 - 0.8 * (i / len(track))
			weighted_color = [weight * c for c in color]
			cv2.circle(annotated_frame, (int(bbox[0, 0]), int(bbox[1, 0])), int(weight * 6), weighted_color, 2) # raw cxy
			cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), weighted_color, 2)
	
	return annotated_frame	