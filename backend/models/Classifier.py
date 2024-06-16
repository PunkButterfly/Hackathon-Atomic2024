from ultralytics import YOLO
import os

class Classifier:
    def __init__(
            self,
            path_to_weights: str,
            weights_name: str = 'classifier_weights.pt'
        ):
        self.model = YOLO(os.path.join(path_to_weights, weights_name))

        
    def predict(self, paths_to_images: list):
        outputs = self.model(paths_to_images, verbose=False)
        
        preds = []
        
        for i, output in enumerate(outputs):
            class_label = output.probs.top1
            conf = output.probs.top1conf.item()
            preds.append(
                {
                    "class_label": class_label,
                    "conf": conf
                })
            
        return preds