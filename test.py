import sys
from util.gamma import test_gamma
from util.clahe import test_clahe, test_self_interpolation

args = sys.argv

# test functions
def run_test_gamma():
	inputs = ['input/video_2.mp4', 'input/video_1.mp4']

	for input_file in inputs:
		test_gamma(input_file, gamma=1.5)

def run_test_clahe():
	# test_self_interpolation(grid_shape=(2, 2), image_size=(6, 6), print_flag=True)
	# test_self_interpolation(grid_shape=(2, 2), image_size=(100, 100), print_flag=False)
	# test_clahe(grid_shape=(2, 2), show_self=True, visualize=True, show_opencv=True, show_difference=True, log_frame_size=10)
	# test_clahe(grid_shape=(3, 3), show_self=True, show_opencv=True, show_difference=True)
	test_clahe(grid_shape=(4, 4), show_self=True, show_opencv=True, show_difference=True)


# command parameters
if args[1] in ['gamma', 'g']:
	run_test_gamma()

if args[1] in ['clahe', 'c']:
	run_test_clahe()