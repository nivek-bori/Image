import logging
from util.config import ByteTrackLogConfig, ByteTrackRenderConfig, ReidConfig
from util.logs import Logger
from tracking import self_byte_track, ultra_byte_track
from util.util import keyboard_quitter

logging.getLogger('pynput').setLevel(logging.ERROR)

# CLI parameters
if __name__ == '__main__':
    import sys

    args = sys.argv

    # bytetrack
    if args[1] in ['bytetrack', 'byte', 'b', 'self', 's']:
        try:
            print('init self byte track')
            log_config = ByteTrackLogConfig()
            render_config = ByteTrackRenderConfig(frame_start=0, frame_end=50)
            reid_config = ReidConfig()
            keyboard_quitter(self_byte_track, cleanup_func=log_config.log_cleanup, input='input/video_1.mp4')
        except Exception as e:
            raise e
        finally:
            print('end self byte track')

            logger = Logger()
            logger.log_timing()

    # ultralytics
    if args[1] in ['ultralytics', 'ultra', 'u']:
        try:
            print('init ultralytics')
            keyboard_quitter(ultra_byte_track)
        except Exception as e:
            raise e
        finally:
            print('end ultralytics')

            logger = Logger()
            logger.log_timing()
