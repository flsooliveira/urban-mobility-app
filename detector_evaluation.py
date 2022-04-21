import cv2
import os
import time
import numpy as np
import pandas as pd

from model.detector import Detector

def compute_error(errors, predicted, ground_truth):
    for class_idx in range(len(errors)):
        if ground_truth[class_idx] > 0:
            error = (abs(ground_truth[class_idx] - predicted[class_idx]) * 100)/ground_truth[class_idx]
        else:
            if predicted[class_idx] == 0:
                error = 0.0
            else:
                error = 10*predicted[class_idx]
                if error > 100:
                    error = 100.0
            
        errors[class_idx].append(error)

classes = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

models = os.listdir("weights")
for model in models:
    det = Detector("yolov5", f"weights/{model}")
    inf_times = []
    errors = [[], [], [], [], [], []]

    sequences = os.listdir("dataset")
    for seq in sequences:
        seq_path = f"dataset/{seq}"
        imgs_names = os.listdir(seq_path)
        print(seq_path)

        with open(f"dataset/{seq}/{imgs_names[-1]}") as f:
            ground_truths = f.readlines()

        for idx, img_name in enumerate(imgs_names[:-1]):
            img_path = f"dataset/{seq}/{img_name}"
            img = cv2.imread(img_path)

            start_ts = time.time()
            bboxes = det.detect(img)
            inf_times.append(time.time() - start_ts)
            
            bboxes_list = list(bboxes.values())

            ground_truth = ground_truths[idx + 1].replace('\n', '').split(' ')
            ground_truth = [int(gt) for gt in ground_truth]
            predicted = [len(bboxes_list[0]), len(bboxes_list[1]), len(bboxes_list[2]),
                         len(bboxes_list[3]), len(bboxes_list[4]), len(bboxes_list[5])]

            compute_error(errors, predicted, ground_truth)
            
            processed_img = img.copy()
            for class_idx in range(len(classes)):
                
                for bbox in bboxes_list[class_idx]:
                    cv2.rectangle(processed_img, (bbox[0], bbox[2]), (bbox[1], bbox[3]), 
                                colors[class_idx], 2)

                    cv2.putText(processed_img, classes[class_idx], (bbox[0], bbox[2] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_idx], 1, cv2.LINE_AA, False)
                                
            model_name = model.replace(".pt", "")
            cv2.imwrite(f"output/{seq}/{model_name}-{img_name}", processed_img)

        mean_inf_time = np.mean(inf_times)
        mean_person_error = np.mean(errors[0])
        mean_bicycles_error = np.mean(errors[1])
        mean_cars_error = np.mean(errors[2])
        mean_motorbike_error = np.mean(errors[3])
        mean_bus_error = np.mean(errors[4])
        mean_truck_error = np.mean(errors[5])

        print(f"\n{model} - {seq}")
        print(f"{mean_inf_time}")
        print(f"{mean_person_error}")
        print(f"{mean_bicycles_error}")
        print(f"{mean_cars_error}")
        print(f"{mean_motorbike_error}")
        print(f"{mean_bus_error}")
        print(f"{mean_truck_error}")

