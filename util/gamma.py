import cv2
import numpy as np
from util.video import video_to_frames

def apply_gamma(image, gamma):
	normalized = image / 255.0
	corrected = np.power(normalized, 1 / gamma) 
	return (corrected * 255.0).astype(np.uint8)

def test_gamma(input_file='input/video_1.mp4', gamma=2):
	frame = next(video_to_frames(input_file))
	
	gamma_frame = apply_gamma(frame, gamma)

	cv2.imshow('frame', frame)
	cv2.waitKey(0)
	cv2.imshow('gamma frame', gamma_frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()