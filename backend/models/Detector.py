from ultralytics import YOLO
import os
from datetime import datetime
from collections import Counter
import numpy as np
from .utils import *

class Detector:
    def __init__(
            self, 
            path_to_weights: str, 
            path_to_tmp: str, 
            weights_name: str = 'detector_weights_v1.pt', 
            confidence_level: float = 0.25, 
            box_intersetor_flag: bool = False,
            iou_treshold: float = 0.8
        ):
        self.model = YOLO(os.path.join(path_to_weights, weights_name))
        self.confidence_level = confidence_level
        self.tmp_path = path_to_tmp
        self.box_intersetor_flag = box_intersetor_flag
        self.iou_treshold = iou_treshold

    def predict(self, paths_to_images: list):
        outputs = self.model.predict(paths_to_images, verbose=False, conf = self.confidence_level)

        preds = []

        for i, output in enumerate(outputs):
            names = [output.names[i] for i in output.boxes.cls.numpy()]
            confs = output.boxes.conf.numpy().tolist()
            coords = output.boxes.xyxy.numpy()

            predict_img_path = f'{self.tmp_path}/tmp_{datetime.now()}.jpg'
            draw_bounding_boxes(paths_to_images[i], predict_img_path, names, coords)

            preds.append(
                {
                    "predict_img_path": predict_img_path,
                    "source_img_path": paths_to_images[i],
                    "names": names,
                    "confs": confs,
                    "coords": coords,
                    "xywhn": output.boxes.xywhn.numpy(),
                    "names_count": dict(Counter(names))
                }
            )

        if not self.box_intersetor_flag:
            return preds
        
        intersected_preds = []

        for i, pred in enumerate(preds):
            indices = non_maximum_suppression(pred["confs"], pred["coords"], iou_threshold=self.iou_treshold)

            intersected_names = np.array(names)[indices]
            intersected_coords = output.boxes.xyxy.numpy()[indices]

            draw_bounding_boxes(pred['source_img_path'], pred['predict_img_path'], intersected_names.tolist(), intersected_coords)

            intersected_preds.append(
                {
                    "predict_img_path": predict_img_path,
                    "source_img_path": paths_to_images[i],
                    "names": intersected_names,
                    "confs": np.array(confs)[indices],
                    "coords": intersected_coords,
                    "xywhn": output.boxes.xywhn.numpy()[indices],
                    "names_count": dict(Counter(intersected_names))
                }
            )

        return intersected_preds