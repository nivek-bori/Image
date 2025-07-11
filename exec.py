import sys
from util.logs import Logger
from tracking import self_byte_track, ultra_byte_track

args = sys.argv
if args[1] in ['bytetrack', 'byte', 'b']:
	try:
		print('init self byte track')
		
		self_byte_track()
	finally:
		print('end self byte track')

		logger = Logger()
		logger.log_timing()

if args[1] in ['ultralytics', 'ultra', 'u']:
	try:
		print('init ultralytics')
		
		ultra_byte_track()
	finally:
		print('end ultralytics')

		logger = Logger()
		logger.log_timing()