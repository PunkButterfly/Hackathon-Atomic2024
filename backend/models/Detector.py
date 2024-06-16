from ultralytics import YOLO
import os
from datetime import datetime
from collections import Counter
import numpy as np

def iou(box1, box2):
    """
    Вычисляет коэффициент перекрытия (IoU) между двумя bounding boxes.
    
    :param box1: Координаты первого бокса [x1, y1, x2, y2]
    :param box2: Координаты второго бокса [x1, y1, x2, y2]
    :return: IoU
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / min(box1_area, box2_area)

def non_maximum_suppression(confs, coords, iou_threshold=0.8):
    """
    Применяет Non-Maximum Suppression (NMS) для устранения перекрывающихся боксов.
    
    :param names: Список имен классов
    :param confs: Список уверенности в обнаружении
    :param coords: Список координат боксов [x1, y1, x2, y2]
    :param iou_threshold: Порог IoU для определения перекрывающихся боксов
    :return: Оставшиеся боксы после применения NMS
    """
    indices = np.argsort(confs)[::-1]
    
    keep_boxes = []
    while len(indices) > 0:
        current_index = indices[0]
        current_box = coords[current_index]
        
        keep_boxes.append(current_index)
        
        iou_scores = np.array([iou(current_box, coords[i]) for i in indices[1:]])
        
        indices = indices[1:][iou_scores <= iou_threshold]
        
    return keep_boxes

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
        # print("Detection...")
        outputs = self.model.predict(paths_to_images, verbose=False, conf = self.confidence_level)

        preds = []

        for i, output in enumerate(outputs):
            names = [output.names[i] for i in output.boxes.cls.numpy()]
            confs = output.boxes.conf.numpy().tolist()

            predict_img_path = output.save(filename=f'{self.tmp_path}/tmp_{datetime.now()}.jpg')

            preds.append(
                {
                    "predict_img_path": predict_img_path,
                    "source_img_path": paths_to_images[i],
                    "names": names,
                    "confs": confs,
                    "coords": output.boxes.xyxy.numpy(),
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
            intersected_preds.append(
                {
                    "predict_img_path": predict_img_path,
                    "source_img_path": paths_to_images[i],
                    "names": intersected_names,
                    "confs": np.array(confs)[indices],
                    "coords": output.boxes.xyxy.numpy()[indices],
                    "xywhn": output.boxes.xywhn.numpy()[indices],
                    "names_count": dict(Counter(intersected_names))
                }
            )

        return intersected_preds