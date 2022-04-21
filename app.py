import argparse
import os
import cv2
import numpy as np
import time

from model.detector import Detector
from model.tracker import Tracker

def main(args):
    detector, tracker = setup(yolo_path=args.yolo_path, 
                              yolo_weights=args.yolo_weights)

    imgs_names = os.listdir(args.dataset_path)
    n_imgs = len(imgs_names) - 1

    start_ts = time.time()
    for frame_idx in range(n_imgs):
        img = acquisition(f"{args.dataset_path}/{imgs_names[0]}")
        del imgs_names[0]

        bboxes = detection(detector=detector,
                           img=img,
                           debug=True)

        tracked_comps = tracking(tracker=tracker,
                                 bboxes=bboxes)           
    
    duration = time.time() - start_ts
    print(f"{n_imgs/duration} FPS")

def setup(yolo_path, yolo_weights):
    detector = Detector(yolo_path=yolo_path, 
                        yolo_weights=yolo_weights)

    tracker = Tracker()

    return detector, tracker

def acquisition(image_path):
    img = cv2.imread(image_path)
    return img

def detection(detector, img, debug):
    bboxes = detector.detect(img)
    bboxes_list = list(bboxes.values())

    if debug:
        processed_img = img.copy()
        classes = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

        for class_idx in range(len(classes)):
            for bbox in bboxes_list[class_idx]:
                cv2.rectangle(processed_img, (bbox[0], bbox[2]), (bbox[1], bbox[3]), 
                              colors[class_idx], 2)

                cv2.putText(processed_img, classes[class_idx], (bbox[0], bbox[2] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_idx], 1, cv2.LINE_AA, False)

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", processed_img)
        cv2.waitKey(0)
    return bboxes_list

def tracking(bboxes, tracker):
    tracked_comps = list()

    for bbox in bboxes:
        if len(bbox) > 0:
            tracked_comps.append(tracker.apply_tracker(np.asarray(bbox)))
        else:
            tracked_comps.append([])
    
    return tracked_comps

def extraction():
    pass

def transmission():
    pass

def storage():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", required=True,
                        help="Path to dataset")

    parser.add_argument("--yolo_path", required=True,
                        help="Path to YOLOv5 weights")

    parser.add_argument("--yolo_weights", required=True,
                        help="Path to YOLOv5 weights")

    args = parser.parse_args()

    main(args)