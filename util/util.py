import cv2
import _thread
import threading
from pynput import keyboard  # Use pynput instead of keyboard


def keyboard_quitter(func, cleanup_func=None, *args, **kwargs):
    keys_pressed = set()

    def moniter_keyb():
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def on_press(key):
        keys_pressed.add(key)

        w_pressed = (
            keyboard.KeyCode(char='w') in keys_pressed
            or keyboard.KeyCode(char='W') in keys_pressed
        )
        shift_pressed = keyboard.Key.shift in keys_pressed

        if w_pressed and shift_pressed:
            print('\nAttempting to force quit...')

            if not cleanup_func is None:
                cleanup_func()

            _thread.interrupt_main()
            
            print('Force quit')

    def on_release(key):
        if key in keys_pressed:
            keys_pressed.remove(key)

    keyb_thread = threading.Thread(target=moniter_keyb)
    keyb_thread.daemon = (
        True  # allow main process to exit even if this thread is running
    )
    keyb_thread.start()

    output = func(*args, **kwargs)  # run provided function, store output

    # wait for keyb_thread to finish
    keyb_thread.join(timeout=0.5)

    print('Process terminated')
    return output # return output