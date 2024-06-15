from Detector import Detector
import os
from tqdm import tqdm
import pandas as pd

def format_output(predicts):
    formated_predicts = {
            'filename' : [],
            'class_id' : [],
            'rel_x' : [],
            'rel_y' : [],
            'width' : [],
            'height' : []
        }

    classes_map = {
        "adj": 0,
        "int": 1,
        "geo": 2,
        "pro": 3,
        "non": 4,
    }
    
    for item in predicts:
        image_name = item['source_img_path'].split('/')[-1]
        for i, cls in enumerate(item['names']):
            parsed_coords = item['xywhn'].tolist()

            formated_predicts['filename'].append(image_name)
            formated_predicts['class_id'].append(classes_map[cls])
            formated_predicts['rel_x'].append(parsed_coords[i][0])
            formated_predicts['rel_y'].append(parsed_coords[i][1])
            formated_predicts['width'].append(parsed_coords[i][2])
            formated_predicts['height'].append(parsed_coords[i][3])
            
    return formated_predicts

def main(
        weights_path='models/weights/', 
        path_to_tmp='./tmp_files', 
        weights_name='detector_weights_v2.pt',
        dataset_path='../dataset',
        submission_file_path='submission.csv'
        ):

    model = Detector(path_to_weights=weights_path, path_to_tmp=path_to_tmp, weights_name=weights_name)

    image_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) if filename.endswith('.jpg')]

    predicts = []
    for img in tqdm(image_paths):
        predicts += model.predict([img])

    pd.DataFrame(format_output(predicts)).to_csv(submission_file_path, index=False, sep=';')

main()