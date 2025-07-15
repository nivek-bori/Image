import cv2
import numpy as np
import matplotlib.pyplot as plt
from util.video import video_to_frames, IMAGE2LAB, LAB2IMAGE


### FUNCTIONS/CLASSES

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

# Due to bilinear interpolation not being able to handle multiple neighbors, I wrote my own implementation
# I did not realize that my implementation would have the same flaw
# Flaw: Although interpolation is smooth when changing neighbors, values being interpolated are not smooth when changing neighbors...
# 		...continued: thus the flaw is not unsmooth interpolation but values changing when neighbors change
# Solution: Do not change neighbors (ie have 2x2 shape) - but this is incomplete
def self_interpolate(image, cdfs, luminance_idx, image_shape, grid_shape, tile_size):
	output = image.copy()

	height, width = image_shape # image
	y_step, x_step = tile_size # num pixels in tile along axis xy

	for y in range(height):
		for x in range(width):
			px = image[y, x, luminance_idx]

			t_y, t_x = y // y_step, x // x_step # tile position
			frac_y, frac_x = (y % y_step) / (y_step - 1), (x % x_step) / (x_step - 1) # xy position in tile
			cdist_y, cdist_x = abs(frac_y - 0.5), abs(frac_x - 0.5) # distance from center

			# neighbor calculation
			if frac_x < 0.5 and frac_y < 0.5: # quadrant one
				# if no tile up-left, current tile is up-left
				shift_y = t_y + (0 if t_y - 1 >= 0 else 1)
				shift_x = t_x + (0 if t_x - 1 >= 0 else 1)

				t00 = (shift_y - 1, shift_x - 1)
				t01 = (shift_y - 1, shift_x)
				t10 = (shift_y, shift_x - 1)
				t11 = (shift_y, shift_x)

				weight_y = cdist_y
				weight_x = cdist_x
			elif frac_x >= 0.5 and frac_y < 0.5:
				# if no tile is up-right, current tile is up-right
				shift_y = t_y + (0 if t_y - 1 >= 0 else 1)
				shift_x = t_x + (0 if t_x + 1 < grid_shape[1] else -1)

				t00 = (shift_y - 1, shift_x)
				t01 = (shift_y - 1, shift_x + 1)
				t10 = (shift_y, shift_x)
				t11 = (shift_y, shift_x + 1)

				weight_y = cdist_y
				weight_x = 1 - cdist_x
			elif frac_x < 0.5 and frac_y >= 0.5:
				# if no tile is down-left, current tile is down-left
				shift_y = t_y + (0 if t_y + 1 < grid_shape[0] else -1)
				shift_x = t_x + (0 if t_x - 1 >= 0 else 1)

				t00 = (shift_y, shift_x - 1)
				t01 = (shift_y, shift_x)
				t10 = (shift_y + 1, shift_x - 1)
				t11 = (shift_y + 1, shift_x)

				weight_y = 1 - cdist_y
				weight_x = cdist_x
			elif frac_x >= 0.5 and frac_y >= 0.5:
				# if no tile is down-right, current tile is down-right
				shift_y = t_y + (0 if t_y + 1 < grid_shape[0] else -1)
				shift_x = t_x + (0 if t_x + 1 < grid_shape[1] else -1)

				t00 = (shift_y, shift_x)
				t01 = (shift_y, shift_x + 1)
				t10 = (shift_y + 1, shift_x)
				t11 = (shift_y + 1, shift_x + 1)

				weight_y = 1 - cdist_y
				weight_x = 1 - cdist_x
			else:
				raise ValueError('grid does not support interpolation')

			# bilinear interpolation
			v00 = cdfs[t00[0], t00[1]][px]
			v01 = cdfs[t01[0], t01[1]][px]
			v10 = cdfs[t10[0], t10[1]][px]
			v11 = cdfs[t11[0], t11[1]][px]
			
			# interpolate
			v0 = v00 * weight_x + v01 * (1 - weight_x)
			v1 = v10 * weight_x + v11 * (1 - weight_x)
			v = v0 * weight_y + v1 * (1 - weight_y)

			output[y, x, luminance_idx] = v

# has issue of smooth interpolation, unsmooth change in values being interpolated
def bilinear_interpolate(image, cdfs, luminance_idx, image_shape, grid_shape, tile_size):
	output = image.copy()

	height, width = image_shape[0], image_shape[1] # image
	grid_shape_y, grid_shape_x = grid_shape[0], grid_shape[1] # num grids along axis xy
	y_step, x_step = tile_size[0], tile_size[1] # num pixels in tile along axis xy

	for y in range(height):
		for x in range(width):
			px = image[y, x, luminance_idx]

			t_y, t_x = int((y - y_step / 2) / y_step), int((x - x_step / 2) / x_step) # tile position
			frac_y, frac_x = (y % y_step) / (y_step - 1), (x % x_step) / (x_step - 1) # xy position in tile

			# four corners use nearest cdf
			if t_y < 0 and t_x < 0:
				output_lum = cdfs[t_y + 1, t_x + 1][px]
			elif t_y < 0 and t_x >= grid_shape_x - 1:
				output_lum = cdfs[t_y + 1, t_x][px]
			elif t_y >= grid_shape_y - 1 and t_x < 0:
				output_lum = cdfs[t_y, t_x + 1][px]
			elif t_y >= grid_shape_y - 1 and t_x >= grid_shape_x - 1:
				output_lum = cdfs[t_y, t_x][px]
			# four borders use two cdfs : linear interpolate
			elif t_y < 0 or t_y >= grid_shape_y - 1:
				t_y = 0 if t_y < 0 else grid_shape_y - 1

				v0 = cdfs[t_y, t_x][px]
				v1 = cdfs[t_y, t_x + 1][px]
				output_lum = (1 - frac_y) * v0 + frac_y * v1
			elif t_x < 0 or t_x >= grid_shape_x - 1:
				t_x = 0 if t_x < 0 else grid_shape_x - 1

				v0 = cdfs[t_y, t_x][px]
				v1 = cdfs[t_y + 1, t_x][px]
				output_lum = (1 - frac_x) * v0 + frac_x * v1
			# inner tiles : bilinear interpolate
			else:
				v00 = cdfs[t_y, t_x][px]
				v01 = cdfs[t_y, t_x + 1][px]
				v10 = cdfs[t_y + 1, t_x][px]
				v11 = cdfs[t_y + 1, t_x + 1][px]
				v0 = (1 - frac_x) * v00 + frac_x * v01
				v1 = (1 - frac_x) * v10 + frac_x * v11
				output_lum = (1 - frac_y) * v0 + frac_y * v1
			
			output[y, x, luminance_idx]  = output_lum


### CLAHE
def apply_self_clahe(image, clip_max_prec, image_type='bgr', grid_shape=(3, 3), visualize_flag=False, log_flag=False):
	lab_image, luminance_idx = IMAGE2LAB(image, image_type)
	height, width = lab_image.shape[:2]

	# grid size
	grid_area = grid_shape[0] * grid_shape[1]
	y_step = int((height + grid_shape[0] - 1) // grid_shape[0]) # ceiling division
	x_step = int((width + grid_shape[1] - 1) // grid_shape[1]) # ceiling division

	clip_max = x_step * y_step * clip_max_prec

	hists = np.zeros(grid_shape, dtype=object) # histograms
	cdfs = np.zeros(grid_shape, dtype=object) # prefix histograms
	clip_totals = np.zeros(grid_shape) # prefix histograms
	areas = np.zeros(grid_shape) # prefix histograms

	# calculate cdf for each tile
	for y_idx in range(height // y_step):
		y1 = y_idx * y_step
		y2 = (y_idx + 1) * y_step
		for x_idx in range(width // x_step):
			x1 = x_idx * x_step
			x2 = (x_idx + 1) * x_step

			# calculation
			tile = lab_image[y1: y2, x1: x2]
			hist, cdf, clip_total = calculate_cdf(tile, clip_max, luminance_idx)

			# storage
			hists[y_idx, x_idx] = hist
			cdfs[y_idx, x_idx] = cdf
			clip_totals[y_idx, x_idx] = clip_total
			areas[y_idx, x_idx] = (y2 - y1) * (x2 - x1)
	
	if visualize_flag:
		plt.figure(figsize=(9 * grid_shape[0], 3 * grid_shape[1]))
		for i in range(grid_area):
			t_y, t_x = i // grid_shape[0], i % grid_shape[1]

			plt.subplot(grid_shape[0], grid_shape[1] * 2, 2 * i + 1)
			plt.plot(range(256), hists[t_y, t_x], alpha=0.7, color='green')
			plt.axhline(clip_max, color='red', linestyle='--', linewidth=1)
			plt.title(f'Tile {i} Histogram', fontsize=8)
			plt.xlabel('Intensity', fontsize=6)
			plt.ylabel('Frequency', fontsize=6)
			plt.tick_params(axis='both', which='major', labelsize=6)
			
			plt.subplot(grid_shape[0], grid_shape[1] * 2, 2 * i + 2)
			plt.plot(range(256), cdfs[t_y, t_x], 'orange')
			plt.title(f'Tile {i} CDF', fontsize=8)
			plt.xlabel('Intensity', fontsize=6)
			plt.ylabel('CDF', fontsize=6)
			plt.tick_params(axis='both', which='major', labelsize=6)
		plt.tight_layout()
		def on_key(_event):
			plt.close()
		plt.gcf().canvas.mpl_connect('key_press_event', on_key)
		plt.show()

	# interpolate each cdf
	output = bilinear_interpolate(image, cdfs, luminance_idx, (height, width), grid_shape, (y_step, x_step))
	# output = self_interpolate(image, cdfs, lumiance_idx, (height, width), grid_shape, (y_step, x_step))

	return LAB2IMAGE(output, image_type)

def apply_opencv_clahe(image, clip_limit=2.0, grid_shape=(8, 8), image_type='bgr'):
    # Convert to LAB if color image
    if len(image.shape) == 3:
        lab_image, luminance_idx = IMAGE2LAB(image, image_type)
        l_channel = lab_image[:, :, luminance_idx]
    else:
        # For grayscale images
        l_channel = image
        lab_image = None
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_shape)
    
    # Apply CLAHE to the L channel
    enhanced_l = clahe.apply(l_channel)
    
    # If original was color image, merge channels back
    if lab_image is not None:
        lab_image[:, :, luminance_idx] = enhanced_l
        return LAB2IMAGE(lab_image, image_type)
    else:
        return enhanced_l


### Testing

# visualize self clahe, opencv clahe, and the difference between self and opencv clahe
def test_clahe(input_file='input/video_1.mp4', grid_shape=(2, 2), visualize=False, log=False, show_self=False, show_opencv=False, show_difference=False, log_frame_size=0):
	# read frame
	frame = next(video_to_frames(input_file))

	# process frames in different styles
	ineffective_frame = (0.5 * frame).astype(np.uint8)
	lab_ineffective_frame = cv2.cvtColor(np.array(IMAGE2LAB(ineffective_frame, 'bgr')[0][:, :, 0]), cv2.COLOR_GRAY2BGR)
	self_clahe_frame = apply_self_clahe(ineffective_frame, 0.2, image_type='bgr', grid_shape=grid_shape, visualize_flag=visualize, log_flag=log)
	opencv_clahe_frame = apply_opencv_clahe(ineffective_frame, 2.0, grid_shape=grid_shape)

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

# visualize interpolation using gray grids
def test_self_interpolation(grid_shape=(2, 2), image_size=(400, 400), print_flag=False):
	height, width = image_size
	grid_rows, grid_cols = grid_shape
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
	
	# ApplyCLAHE with the specified grid
	img_lum = np.array(IMAGE2LAB(img, 'bgr')[0][:, :, 0])
	img_bgr = cv2.cvtColor(img_lum, cv2.COLOR_GRAY2BGR)
	clahe_img = cv2.cvtColor(apply_self_clahe(img_bgr, clip_max_prec=0.8, image_type='bgr', grid_shape=grid_shape, visualize_flag=False), cv2.COLOR_BGR2GRAY)
	
	# Display results
	all_imgs = np.hstack([img_lum, clahe_img])
	print(img_lum, '\n', clahe_img)
	cv2.imshow(f'{grid_rows}x{grid_cols}', all_imgs)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	return img, clahe_img