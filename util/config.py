# ByteTrack Configs

class ByteTrackLogConfig:
    def __init__(
        self,
        all_logs=None,
        auto_play=True,
        show_bool=True,
        log_frame_info=True,
        log_results_info=False,
        temporary_frame_info=True,
        log_high_conf_matching=False,
        log_low_conf_matching=False,
    ):
        if all_logs is True:
            self.auto_play = True
            self.show_bool = True
            self.log_frame_info = True
            self.log_results_info = True
            self.temporary_frame_info = True
            self.log_high_conf_matching = True
            self.log_low_conf_matching = True
        elif all_logs is False:
            self.auto_play = False
            self.show_bool = False
            self.log_frame_info = False
            self.log_results_info = False
            self.temporary_frame_info = False
            self.log_high_conf_matching = False
            self.log_low_conf_matching = False
        else:
            self.auto_play = auto_play
            self.show_bool = show_bool
            self.log_frame_info = log_frame_info
            self.log_results_info = log_results_info
            self.temporary_frame_info = temporary_frame_info
            self.log_high_conf_matching = log_high_conf_matching
            self.log_low_conf_matching = log_low_conf_matching

    def log_cleanup(self):
        import cv2

        if self.log_frame_info and self.temporary_frame_info:  # log cleanup
            print(60 * ' ', end='\r') # clear line

        cv2.destroyAllWindows()  # rendering cleanup


class ByteTrackVideoConfig:
    def __init__(self, required_tracklet_age=0, frame_start=0, frame_end=0):
        self.required_tracklet_age = required_tracklet_age # minimum tracklet age before render
        self.frame_start = frame_start  # start of frames to process
        self.frame_end = frame_end  # end of frames to process, 0 for unset


# create the reid config class
class ReidConfig:
    def __init__(self, type='ave', shape=(512, ), mult=1.5, max_lookback=3):
        self.type = type  # reid lookback type
        self.shape = shape  # reid output shape
        self.mult = mult  # multiplier, only for 'time' reid
        self.max_lookback = max_lookback  # how many frames reid pulls features from

# EVALUATE MOT20 CONFIG

class MOT20VideoConfig:
    def __init__(self, frame_start=0, frame_end=0):
        self.frame_start = frame_start  # start of frames to process
        self.frame_end = frame_end  # end of frames to process, 0 for unset