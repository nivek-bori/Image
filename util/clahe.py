import numpy as np
import cv2  # TODO: REMOVE
from video import video_to_frames  # TODO: REMOVE
import random # TODO: REMOVE
import matplotlib.pyplot as plt # TODO: REMOVE


# Functions/Classes
def RGB2LAB(rgb_image):
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    return lab_image, 0
def LAB2RGB(lab_image):
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return rgb_image

def BGR2LAB(bgr_image):
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    return lab_image, 0
def LAB2BGR(lab_image):
    bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    return bgr_image

def IMAGE2LAB(image, image_type='bgr'):
	if image_type == 'bgr':
		return BGR2LAB(image)
	if image_type == 'rgb':
		return RGB2LAB(image)
	raise ValueError('Image type not supported')
def LAB2IMAGE(image, image_type='bgr'):
	if image_type == 'bgr':
		return LAB2BGR(image)
	if image_type == 'rgb':
		return LAB2RGB(image)
	raise ValueError('Image type not supported')

def tile_POS2IDX(num_grids, y, x):
	return y * num_grids[1] + x

def calculate_cdf(image, clip_max, luminance_idx=0):
	hist = np.bincount(image[:, :, luminance_idx].flatten(), minlength=256) # calculate histogram
	
	clip_total = np.sum(np.maximum(0, hist - clip_max)) # clip_total is sum of excess values
	redistribute_per_bin = clip_total / 256 # calculate redistribution amount

	hist = np.minimum(hist, max(0, clip_max - redistribute_per_bin)) # clip histogram
	hist = hist + redistribute_per_bin # add in redistribution

	cdf = np.cumsum(hist) # calcualte cdf
	if cdf[-1] > 0:
		cdf = cdf * (255 / cdf[-1])

	return hist, cdf, clip_total

def interpolate_2x2(cdfs, pos, num_grids, grid_shape, px_strength, log_flag=False):
	# # tile pos
	t_y = pos[0] // grid_shape[0]
	t_x = pos[1] // grid_shape[1]

    # in grid percent
	frac_y = (pos[0] % grid_shape[0]) / (grid_shape[0] - 1)
	frac_x = (pos[1] % grid_shape[1]) / (grid_shape[1] - 1)
	cdist_y = abs(frac_y - 0.5)
	cdist_x = abs(frac_x - 0.5)

	# neighbor calculation
	if frac_x < 0.5 and frac_y < 0.5: # quadrant one
		shift_y = t_y + (0 if t_y - 1 >= 0 else 1)
		shift_x = t_x + (0 if t_x - 1 >= 0 else 1)

		t00 = (shift_y - 1, shift_x - 1)
		t01 = (shift_y - 1, shift_x)
		t10 = (shift_y, shift_x - 1)
		t11 = (shift_y, shift_x)

		weight_y = cdist_y
		weight_x = cdist_x
	elif frac_x >= 0.5 and frac_y < 0.5:
		shift_y = t_y + (0 if t_y - 1 >= 0 else 1)
		shift_x = t_x + (0 if t_x + 1 < num_grids[1] else -1)

		t00 = (shift_y - 1, shift_x)
		t01 = (shift_y - 1, shift_x + 1)
		t10 = (shift_y, shift_x)
		t11 = (shift_y, shift_x + 1)

		weight_y = cdist_y
		weight_x = 1 - cdist_x
	elif frac_x < 0.5 and frac_y >= 0.5:
		shift_y = t_y + (0 if t_y + 1 < num_grids[0] else -1)
		shift_x = t_x + (0 if t_x - 1 >= 0 else 1)

		t00 = (shift_y, shift_x - 1)
		t01 = (shift_y, shift_x)
		t10 = (shift_y + 1, shift_x - 1)
		t11 = (shift_y + 1, shift_x)

		weight_y = 1 - cdist_y
		weight_x = cdist_x
	elif frac_x >= 0.5 and frac_y >= 0.5:
		shift_y = t_y + (0 if t_y + 1 < num_grids[0] else -1)
		shift_x = t_x + (0 if t_x + 1 < num_grids[1] else -1)

		t00 = (shift_y, shift_x)
		t01 = (shift_y, shift_x + 1)
		t10 = (shift_y + 1, shift_x)
		t11 = (shift_y + 1, shift_x + 1)

		weight_y = 1 - cdist_y
		weight_x = 1 - cdist_x
	else:
		raise ValueError('grid does not support interpolation')
    
	t00 = tile_POS2IDX(num_grids, t00[0], t00[1])
	t01 = tile_POS2IDX(num_grids, t01[0], t01[1])
	t10 = tile_POS2IDX(num_grids, t10[0], t10[1])
	t11 = tile_POS2IDX(num_grids, t11[0], t11[1])

    # bilinear interpolation
	v00 = cdfs[t00][px_strength]
	v01 = cdfs[t01][px_strength]
	v10 = cdfs[t10][px_strength]
	v11 = cdfs[t11][px_strength]
    
    # Interpolate
	v0 = v00 * weight_x + v01 * (1 - weight_x)
	v1 = v10 * weight_x + v11 * (1 - weight_x)
	v = v0 * weight_y + v1 * (1 - weight_y)

	if log_flag and random.random() < 1:
		print(
			f'XX{tile_POS2IDX(num_grids, t_y, t_x)}', pos[0] % grid_shape[0], pos[1] % grid_shape[1], ' - ',
			weight_y, weight_x, ' - ',
			f"{v00:.4g}", f"{v01:.4g}", f"{v10:.4g}", f"{v11:.4g}",
			'       = ', f"{v:.4g}",
		)
		# print(
		# 	tile_POS2IDX(num_grids, t_y, t_x), px_strength, ' - ', 
		# 	t00, t01, t10, t11, ' - ', 
		# 	f"{v00:.4g}", f"{v01:.4g}", f"{v10:.4g}", f"{v11:.4g}", ' - ', 
		# 	f"{v0:.4g}", f"{v1:.4g}", '       = ', f"{v:.4g}",
		# )

	if log_flag:
		return cdfs[tile_POS2IDX(num_grids, t_y, t_x)][px_strength]
	return v


# CLAHE
def apply_self_clahe(image, clip_max_prec, image_type='bgr', num_grids=(3, 3), visualize_flag=False, log_flag=False):
	lab_image, luminance_idx = IMAGE2LAB(image, image_type)
	height, width = lab_image.shape[:2]

	# grid size
	grid_area = num_grids[0] * num_grids[1]
	y_step = int((height + num_grids[0] - 1) // num_grids[0]) # ceiling division
	x_step = int((width + num_grids[1] - 1) // num_grids[1]) # ceiling division

	clip_max = x_step * y_step * clip_max_prec

	hists = np.zeros((grid_area), dtype=object) # histograms
	cdfs = np.zeros((grid_area), dtype=object) # prefix histograms
	clip_totals = np.zeros((grid_area)) # prefix histograms
	areas = np.zeros((grid_area)) # prefix histograms

	# preprocessing (per grid values)
	tile_idx = 0
	for y1 in range(0, height, y_step):
		y2 = min(height, y1 + y_step)
		for x1 in range(0, width, x_step):
			x2 = min(width, x1 + x_step)

			# calculation
			tile = lab_image[y1: y2, x1: x2]
			hist, cdf, clip_total = calculate_cdf(tile, clip_max, luminance_idx)

			# storage
			hists[tile_idx] = hist
			cdfs[tile_idx] = cdf
			clip_totals[tile_idx] = clip_total
			areas[tile_idx] = (y2 - y1) * (x2 - x1)

			tile_idx += 1
	
	if visualize_flag:
		plt.figure(figsize=(9 * num_grids[0], 3 * num_grids[1]))
		for i in range(grid_area):
			plt.subplot(num_grids[0], num_grids[1] * 2, 2 * i + 1)
			plt.plot(range(256), hists[i], alpha=0.7, color='green')
			plt.axhline(clip_max, color='red', linestyle='--', linewidth=1)
			plt.title(f'Tile {i} Histogram', fontsize=8)
			plt.xlabel('Intensity', fontsize=6)
			plt.ylabel('Frequency', fontsize=6)
			plt.tick_params(axis='both', which='major', labelsize=6)
			
			plt.subplot(num_grids[0], num_grids[1] * 2, 2 * i + 2)
			plt.plot(range(256), cdfs[i], 'orange')
			plt.title(f'Tile {i} CDF', fontsize=8)
			plt.xlabel('Intensity', fontsize=6)
			plt.ylabel('CDF', fontsize=6)
			plt.tick_params(axis='both', which='major', labelsize=6)
		plt.tight_layout()
		def on_key(_event):
			plt.close()
		plt.gcf().canvas.mpl_connect('key_press_event', on_key)
		plt.show()

	# applying processing to each pixel
	output = lab_image.copy()

	tile_idx = 0
	for y, row in enumerate(lab_image):
		for x, px in enumerate(row):
			tile_idx = num_grids[1] * (y // y_step) + (x // x_step)

			# modify original lum value
			px_strength = px[luminance_idx]
			output[y, x, luminance_idx] = interpolate_2x2(cdfs, (y, x), num_grids, (y_step, x_step), px_strength, log_flag=log_flag)
			
	return LAB2IMAGE(output, image_type)

def apply_opencv_clahe(image, clip_limit=2.0, num_grids=(8, 8), image_type='bgr'):
    # Convert to LAB if color image
    if len(image.shape) == 3:
        lab_image, luminance_idx = IMAGE2LAB(image, image_type)
        l_channel = lab_image[:, :, luminance_idx]
    else:
        # For grayscale images
        l_channel = image
        lab_image = None
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=num_grids)
    
    # Apply CLAHE to the L channel
    enhanced_l = clahe.apply(l_channel)
    
    # If original was color image, merge channels back
    if lab_image is not None:
        lab_image[:, :, luminance_idx] = enhanced_l
        return LAB2IMAGE(lab_image, image_type)
    else:
        return enhanced_l

# Testing
def test_clahe(input_file='input/video_1.mp4', num_grids=(2, 2), visualize=False, log=False, show_self=False, show_opencv=False, show_difference=False, log_frame_size=0):
	# read frame
	frame = next(video_to_frames(input_file))

	# process frames in different styles
	ineffective_frame = (0.5 * frame).astype(np.uint8)
	lab_ineffective_frame = cv2.cvtColor(np.array(IMAGE2LAB(ineffective_frame, 'bgr')[0][:, :, 0]), cv2.COLOR_GRAY2BGR)
	self_clahe_frame = apply_self_clahe(ineffective_frame, 0.2, image_type='bgr', num_grids=num_grids, visualize_flag=visualize, log_flag=log)
	opencv_clahe_frame = apply_opencv_clahe(ineffective_frame, 2.0, num_grids=num_grids)

	# show frames
	cv2.imshow('ineffective frame', ineffective_frame)
	cv2.waitKey(0)
	cv2.imshow('lab ineffective frame', lab_ineffective_frame)
	cv2.waitKey(0)

	if show_self:
		if log_frame_size > 0:
			print('self clahe frame:\n', self_clahe_frame[0: log_frame_size, 0: log_frame_size])
		cv2.imshow('self clahe frame', self_clahe_frame)
		cv2.waitKey(0)
	if show_opencv:
		if log_frame_size > 0:
			print('opencv clahe frame:\n', opencv_clahe_frame[0: log_frame_size, 0: log_frame_size])
		cv2.imshow('opencv clahe frame', opencv_clahe_frame)
		cv2.waitKey(0)
	if show_difference:
		diff = abs(opencv_clahe_frame - self_clahe_frame)
		if log_frame_size > 0:
			print('difference frame:\n', diff[0: log_frame_size, 0: log_frame_size])
		cv2.imshow('difference frame', diff)
		cv2.waitKey(0)
		
	cv2.destroyAllWindows()

def test_self_interpolation(num_grids=(2, 2), image_size=(400, 400), print_flag=False):
	height, width = image_size
	grid_rows, grid_cols = num_grids
	img = np.zeros((height, width, 3), dtype=np.uint8)
	
	# Calculate tile dimensions
	tile_height = height // grid_rows
	tile_width = width // grid_cols
	
	# Generate different grey values for each tile
	total_tiles = grid_rows * grid_cols
	grey_values = np.linspace(30, 220, total_tiles, dtype=np.uint8)  # Spread from dark to light

	# Fill each tile with a different grey value
	tile_idx = 0
	for row in range(grid_rows):
		for col in range(grid_cols):
			y1 = row * tile_height
			y2 = (row + 1) * tile_height if row < grid_rows - 1 else height
			x1 = col * tile_width
			x2 = (col + 1) * tile_width if col < grid_cols - 1 else width
			
			img[y1:y2, x1:x2, :] = [grey_values[tile_idx]] * 3
			tile_idx += 1
	
	# Apply CLAHE with the specified grid
	img_lum = np.array(IMAGE2LAB(img, 'bgr')[0][:, :, 0])
	img_bgr = cv2.cvtColor(img_lum, cv2.COLOR_GRAY2BGR)
	cdf_image = cv2.cvtColor(apply_self_clahe(img_bgr, clip_max_prec=0.8, image_type='bgr', num_grids=num_grids, visualize=False, log_flag=True), cv2.COLOR_BGR2GRAY)
	clahe_img = cv2.cvtColor(apply_self_clahe(img_bgr, clip_max_prec=0.8, image_type='bgr', num_grids=num_grids, visualize=False), cv2.COLOR_BGR2GRAY)
	
	# Display results
	all_imgs = np.hstack([img_lum, cdf_image, clahe_img])
	print(img_lum, '\n', cdf_image, '\n', clahe_img[1 : -1, 1 : -1])
	cv2.imshow(f'{grid_rows}x{grid_cols}', all_imgs)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	return img, clahe_img

if __name__ == "__main__":
	# test_self_interpolation(num_grids=(2, 2), image_size=(6, 6), print_flag=True)
	# test_self_interpolation(num_grids=(2, 2), image_size=(100, 100), print_flag=False)
	test_clahe(num_grids=(2, 2), show_self=True, show_opencv=True, show_difference=True, log_frame_size=10)
	# test_clahe(num_grids=(3, 3), show_self=True, show_opencv=True, show_difference=True)
	# test_clahe(num_grids=(4, 4), show_self=True, show_opencv=True, show_difference=True)