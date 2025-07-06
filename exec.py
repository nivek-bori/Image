import sys
from tracking import self_byte_track, ultra_byte_track

args = sys.argv
if args[1] in ['bytetrack', 'byte', 'b']:
	print('init self byte track')
	self_byte_track()
	print('end self byte track')

if args[1]in ['ultralytics', 'ultra', 'u']:
	print('init ultralytics')
	ultra_byte_track()
	print('end ultralytics')