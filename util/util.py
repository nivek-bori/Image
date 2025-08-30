import cv2
import _thread
import threading
import numpy as np
from pynput import keyboard

from util.matching import calculate_iou

# MULTIPLE USE FUNCTIONS

class MOTDataFrame:
    def __init__(self, frame, ids, xywhs):
        self.frame = frame
        self.ids = ids
        self.xywhs = np.array(xywhs)

    def __str__(self):
        return f'MOTDataFrame(frame: {self.frame}, ids: {self.ids}, xywhs: [{len(self.xywhs)} bboxes)]'

    def __repr__(self):
        return self.__str__()

        
def calculate_cost_mat(gt_xywh, ts_xywh):
    # format ground truth and tracklet xywh arrays into 2D mat
    row_xywh = gt_xywh[:, np.newaxis, :]
    col_xywh = ts_xywh[np.newaxis, :, :]

    # calculate iou
    iou = calculate_iou(row_xywh, col_xywh)

    # calculate cost
    cost = np.full_like(iou, fill_value=1.0)
    cost += 1 - iou
    return cost


def slice_generator(gen, start, end):
    try:
        for _iter in range(start):
            next(gen)
    except StopIteration:
        return

    for i, item in enumerate(gen, start=start):
        if end > 0 and i < end:
            yield item
        else:
            return


def wait_key_press():
    # Create window and force focus
    cv2.namedWindow('Press any key to continue...', cv2.WINDOW_NORMAL)
    cv2.imshow('Press any key to continue...', np.zeros((200, 400, 3)))
    # cv2.setWindowProperty('Test', cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    # cv2.setWindowProperty('Test', cv2.WND_PROP_TOPMOST, 0)

    print('Press any key to continue...')
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()


def keyboard_quitter(func, cleanup_func=None, *args, **kwargs):
    keys_pressed = set()

    def moniter_keyb():
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def on_press(key):
        keys_pressed.add(key)

        w_pressed = keyboard.KeyCode(char='w') in keys_pressed or keyboard.KeyCode(char='W') in keys_pressed
        shift_pressed = keyboard.Key.shift in keys_pressed

        if w_pressed and shift_pressed:
            print('\nAttempting to force quit...')

            if not cleanup_func is None:
                cleanup_func()

            _thread.interrupt_main()

    def on_release(key):
        if key in keys_pressed:
            keys_pressed.remove(key)

    keyb_thread = threading.Thread(target=moniter_keyb)
    keyb_thread.daemon = True # allow main process to exit even if this thread is running
    keyb_thread.start()

    try:
        if func is None:
            raise Exception('Function provided returned NoneType. Ensure function is passed, not called')
        output = func(*args, **kwargs)  # run provided function, store output
    except Exception as e:
        raise
    # wait for keyb_thread to finish
    keyb_thread.join(timeout=0.5)

    print('Process terminated')
    return output # return output