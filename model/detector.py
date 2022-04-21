import torch
import pandas as pd

class Detector:
    """
    This class create car detector.

    ...

    Attributes
    ----------
    model : Torch
        YOLOv5 model

    Methods
    -------
    detect(image)
        detects cars in an image

    """

    def __init__(self, yolo_path, yolo_weights):
        #Loads YOLO model
        self.model = torch.hub.load(yolo_path, 'custom', path=yolo_weights, source='local')

        self.classes = None
        if "coco" in yolo_weights:
            self.classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
        elif "VOC" in yolo_weights:
            self.classes = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
        elif "VisDrone" in yolo_weights:
            self.classes = ["pedestrian", "bicycle", "car", "motor", "bus", "truck"]

    def detect(self, image):

        """
        Detects caras in an image

        Args:
            image (np.array): urban scenario iamge

        Return:
            (list): detections bounding boxes

        """

        #Radar image model inference
        detection_results = self.model(image)

        #Gets detected objects data
        df = detection_results.pandas().xyxy[0]
        filtered_df = df[df["name"] == self.classes[0]]

        #Filter detected objects
        for class_name in self.classes[1:]:
            filtered_df = pd.concat([filtered_df, df[df["name"] == class_name]], axis=0)

        #Gets objects indexes
        objects_idx = filtered_df.index

        #Cars bounding boxes list
        bounding_boxes = {
            self.classes[0]: [],
            self.classes[1]: [],
            self.classes[2]: [],
            self.classes[3]: [],
            self.classes[4]: [],
            self.classes[5]: []
        }

        #Loop over all detected cars by index
        for idx in objects_idx:
            
            #Gets detection bounding box
            if filtered_df['confidence'][idx] > 0.20:
                class_name = filtered_df['name'][idx]
                x_min, x_max = int(filtered_df['xmin'][idx]), int(filtered_df['xmax'][idx])
                y_min, y_max = int(filtered_df['ymin'][idx]), int(filtered_df['ymax'][idx])

                #Inserts current bounding box in list
                bounding_boxes[class_name].append((x_min, x_max, y_min, y_max))

        return bounding_boxes