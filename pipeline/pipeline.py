import os
import time


os.system("python train.py --data face_detection.yaml --cfg yolov5x.yaml --weights '' --batch-size 8 --epochs 2")
time.sleep(4)

os.system("python train.py --data face_detection.yaml --cfg yolov5s.yaml --weights '' --batch-size 8 --epochs 2")
time.sleep(4)

os.system("python train.py --data face_detection.yaml --cfg yolov5l.yaml --weights '' --batch-size 8 --epochs 2")
time.sleep(4)

os.system("python train.py --data face_detection.yaml --cfg yolov5m.yaml --weights '' --batch-size 8 --epochs 2")
time.sleep(4)

os.system("python train.py --data face_detection.yaml --cfg yolov5n.yaml --weights '' --batch-size 8 --epochs 2")
time.sleep(4)

os.system("python train.py --data face_detection.yaml --cfg yolov5l6.yaml --weights '' --batch-size 8 --epochs 2")
time.sleep(4)

os.system("python train.py --data face_detection.yaml --cfg yolov5m6.yaml --weights '' --batch-size 8 --epochs 2")
time.sleep(4)

os.system("python train.py --data face_detection.yaml --cfg yolov5n6.yaml --weights '' --batch-size 8 --epochs 2")
time.sleep(4)

os.system("python train.py --data face_detection.yaml --cfg yolov5s6.yaml --weights '' --batch-size 8 --epochs 2")
time.sleep(4)
