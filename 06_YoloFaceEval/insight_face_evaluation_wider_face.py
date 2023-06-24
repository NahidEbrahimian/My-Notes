import cv2
import os
import argparse
import numpy as np
import timeit
from tqdm import tqdm
import pickle
import random

from face_detector import FaceDetector


# ARGPARSE
parser = argparse.ArgumentParser(description='Process some aurguments')
parser.add_argument('--folder_address', default="./widerface/val/images", type=str, help='Checkpoint folder address')
parser.add_argument('--model_address', default="./model.onnx", type=str, help='Model address')
parser.add_argument('--result_file', default="detection_result_alldata_bs1.txt", type=str, help='Result text file address')
args = parser.parse_args()

# ARGUMENTS
dataset_folder = args.folder_address
result_file_address = args.result_file
detection_folder = "./widerface-evaluation-master/my_detections/"
image_results_folder = "./wider_face_random_results/"
det_size = 640
det_thresh = 0.02
inference_batch_size = 1  # not used yet
write_some_images = False

if not os.path.exists(detection_folder):
    os.mkdir(detection_folder)
else:
    os.system("rm -r " + detection_folder)
    os.mkdir(detection_folder)

# CREATE MODEL
detection_model = FaceDetector(model_address=args.model_address, det_size=det_size, det_thresh=det_thresh, gpu_load=True)

# EVALUATE WIDER-FACE DATASET
start_time = timeit.default_timer()
image_counter = 0
total_inference_time = 0.0
for folder_name in tqdm(os.listdir(dataset_folder), bar_format='Step:{n_fmt}/{total_fmt} |{bar:30}|'):
    for image_name in os.listdir(os.path.join(dataset_folder, folder_name)):
        # print(image_name)
        
        # LOAD IMAGE
        image_address = os.path.join(dataset_folder, folder_name, image_name)
        img = cv2.imread(image_address)
        
        # MODEL INFERENCE
        # print(img.shape)
        faces, inference_time = detection_model.inference(img, dynamic=False)  # change to True for dynamic inference, 
        total_inference_time += inference_time
        
        if not os.path.exists(os.path.join(detection_folder, folder_name)):
            os.mkdir(os.path.join(detection_folder, folder_name))     
          
        wider_face_predictions = image_name[:-4] + "\n" + str(len(faces)) + "\n"
        for index, face in enumerate(faces):
            bbox = face.bbox
            x_min = bbox[0]
            y_min = bbox[1]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            score = face.score
                    
            wider_face_predictions = wider_face_predictions + str(x_min) + " " + str(y_min) + " " + str(width) + " " + str(height) + " " + str(score) + "\n"

        with open(os.path.join(detection_folder, folder_name, image_name[:-3] + 'txt'), "w+") as txt_file:
            txt_file.write(wider_face_predictions)
        
        image_counter += 1

print("FPS = {:1.2f}".format(image_counter / total_inference_time))
os.system("python widerface-evaluation-master/evaluation.py -p " + detection_folder + " -g ./widerface-evaluation-master/ground_truth/ -r " + result_file_address)
# os.system("rm -r " + detection_folder)
