from ultralytics import YOLO
import os
from datetime import datetime
from collections import Counter

class Detector:
    def __init__(self, path_to_weights: str, path_to_tmp: str, weights_name: str = 'detector_weights_v1.pt', confidence_level: float = 0.001):
        self.model = YOLO(os.path.join(path_to_weights, weights_name))
        self.model.conf = confidence_level
        self.tmp_path = path_to_tmp

    def predict(self, paths_to_images: list):
        # print("Detection...")
        outputs = self.model(paths_to_images, verbose=False)

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
            
        return preds