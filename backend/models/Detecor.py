from ultralytics import YOLO
import os
from datetime import datetime

class Detector:
    def __init__(self, path_to_weights: str, path_to_tmp: str, weights_name: str = 'detector_weights_v1.pt', confidence_level: float = 0.25):
        self.model = YOLO(os.path.join(path_to_weights, weights_name))
        self.model.conf = confidence_level
        self.tmp_path = path_to_tmp

    def predict(self, paths_to_images: list):
        print("Detection...")
        outputs = self.model(paths_to_images)

        preds = []

        for output in outputs:
            names = [output.names[i] for i in output.boxes.cls.numpy()]
            confs = output.boxes.conf.numpy().tolist()

            predict_img_path = output.save(filename=f'{self.tmp_path}tmp_{datetime.now()}.jpg')

            preds.append(
                {
                    "predict_img_path": predict_img_path,
                    "names": names,
                    "confs": confs,
                    "coords": output.boxes.xyxy.numpy(),
                    "xywhn": output.boxes.xywhn.numpy()
                }
            )
            
        return preds