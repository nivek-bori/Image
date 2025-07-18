import cv2
import numpy as np


### VIDEO


def video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield np.array(frame)

    cap.release()


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    return frame_count / fps if fps > 0 else 0


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    return fps


def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    return frame_count


### IMAGE PROCESSING


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
