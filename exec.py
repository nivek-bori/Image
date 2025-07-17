import sys
import time
import signal
import _thread
import logging
import threading
from pynput import keyboard # Use pynput instead of keyboard
from util.logs import Logger
from tracking import self_byte_track, ultra_byte_track
logging.getLogger('pynput').setLevel(logging.ERROR)


# Keyboard quitter
def keyboard_quitter(func, end_func):
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
			_thread.interrupt_main()
			print('Force quit')

	def on_release(key):
		if key in keys_pressed:
			keys_pressed.remove(key)

	keyb_thread = threading.Thread(target=moniter_keyb)
	keyb_thread.daemon = True # allow main process to exit even if this thread is running
	keyb_thread.start()

	func() # run provided function

	# wait for keyb_thread to finish
	keyb_thread.join(timeout=0.5)
	print('Process terminated')

	# end_func() # run provided end function


# CLI Arguement to Function Mapping
args = sys.argv
if args[1] in ['bytetrack', 'byte', 'b', 'self', 's']:
	def end_func():
		print('end self byte track')

		logger = Logger()
		# logger.log_timing()

	try:
		print('init self byte track')
		
		keyboard_quitter(self_byte_track, end_func)
	finally:
		end_func()

if args[1] in ['ultralytics', 'ultra', 'u']:
	def end_func():
		print('end ultralytics')

		logger = Logger()
		logger.log_timing()

	try:
		print('init ultralytics')
		
		keyboard_quitter(ultra_byte_track, end_func)
	finally:
		end_func()