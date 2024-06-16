from models.Classifier import Classifier
from models.Detector import Detector
from PIL import Image
import os

class Pipeline:
    def __init__(
            self, 
            path_to_weights: str, 
            path_to_tmp: str, 
            detector_weights_name: str = 'detector_weights_v1.pt', 
            confidence_level: float = 0.25, 
            box_intersetor_flag: bool = False,
            iou_treshold: float = 0.8,
            classifier_weights_name: str = 'classifier_weights_v1.pt',
            ):
        
        self.detector = Detector(
            path_to_weights,
            path_to_tmp,
            detector_weights_name,
            confidence_level,
            box_intersetor_flag,
            iou_treshold
            )
        
        # self.classifier = Classifier(
        #     path_to_weights, classifier_weights_name
        #     )
        
        # path_to_parts_tmp = os.path.join(path_to_tmp, "tmp_classification")

        # if not os.path.exists(path_to_parts_tmp):
        #     os.mkdir(path_to_parts_tmp)

        # self.path_to_parts_tmp = path_to_parts_tmp
        
    def forward(self, paths_to_images):
        detector_preds = self.detector.predict(paths_to_images)[0]
        
        # with Image.open(detector_preds['predict_img_path']) as img:
        #     cropped_img_paths = []
        #     for idx, coords in enumerate(detector_preds['coords']):
        #         cropped_img = img.crop((coords[0], coords[1], coords[2], coords[3]))

        #         cur_part_path = os.path.join(self.path_to_parts_tmp, str(idx) + ".jpg")

        #         cropped_img.save(cur_part_path)
        #         cropped_img_paths.append(cur_part_path)

        #     classifier_preds = self.classifier.predict(self.path_to_parts_tmp)

        #     print(classifier_preds)

        #     for cropped_img_path in cropped_img_paths:
        #         os.remove(cropped_img_path)
        
        # detector_preds['class_preds'] = classifier_preds
        
        return detector_preds