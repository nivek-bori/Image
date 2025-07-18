from ultralytics import YOLO

model = YOLO('models/yolo11n.pt')

results = model('input/video_1B.mp4', show=True, save=False)
