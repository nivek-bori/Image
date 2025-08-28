PIPELINE:
preprocess frame:
	input(frame) -> gamma correction -> clahe -> output(processed frame)

detect objects using yolo model:
	(method 1) input(processed frame) -> yolo model -> output(detections)
	(method 2) input(dataset ground truth) -> output(detections)

classify detections based on confidence & calculate reid in batch
	input(detections) -> reid model -> output(detections with reid)
	input(detections with reid) -> confidence classification -> output(high confidence detections with reid, low confidence detections with reid)

match high conf detections w/ previous tracklets predictions
	input(tracklets) -> kalman filter -> output(tracklet prediction)
	input(detection, tracklet prediction) -> hungarian matching using cost = f(IoU, age) -> output(matched, unmatched)

match low conf detections w/ lost tracklets
	input(lost tracklets) -> kalman filter output(lost tracklet predictions)
	intput(detections, lost tracklet predictions) -> hungarian matching using cost = f(IoU, age) -> output(matched, unmatched)

update states

visualize

TECHNIQUES IMPLEMENTED:
1. Gamma correction
	a. Purpose: Preprocessing
	b. Route: util/gamma.py/function apply_gamma()
2. CLAHE
	a. Purpose: Preprocessing
	b. Route: util/clahe.py/function apply_opencv_clahe()
3. YOLO
	a. Purpose: Object detection
	b. Route: util/load_model.py/function load_yolo_model()
4. Reid
	a. Purpose: Object identification
	b. Route: util/load_model.py/function load_reid_model()
5. Kalman Filter - 
	a. Purpose: ByteTrack tracklet prediction
	b. Route: util/kalman_filter.py/class KalmanFilter()
6. Greedy Matching
	a. Purpose: ByteTrack matching tracklet to detection
	b. Route: util/matching.py/greedy_match()

FILES:
Route: exec.py x
	Runs tracking code
	x specifies which tracking algorithm

Route: test.py x
	Runs test code
	x specifies which test

Route: tracking.py
	Tracking code - self and ultralytics implementation

Route: util
	All utilities