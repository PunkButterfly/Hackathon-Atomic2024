from Detector import Detector
import os
from tqdm import tqdm
import pandas as pd

def load_annotations(txt_file):
    annotations = []
    with open(txt_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            rel_x = float(parts[1])
            rel_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            annotations.append((class_id, rel_x, rel_y, width, height))
    return annotations

def format_output(predicts, dataset_path='', for_public=False):
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

    if for_public:
        for filename in os.listdir(dataset_path):
            if filename.endswith('.txt') and not filename.startswith('5 '):
                annotations = load_annotations(os.path.join(dataset_path, filename))
                for annot in annotations:
                    formated_predicts['filename'].append(filename.split('.')[0] + '.jpg')
                    formated_predicts['class_id'].append(annot[0])
                    formated_predicts['rel_x'].append(annot[1])
                    formated_predicts['rel_y'].append(annot[2])
                    formated_predicts['width'].append(annot[3])
                    formated_predicts['height'].append(annot[4])
            
    return formated_predicts

def main(
        weights_path='models/weights/', 
        path_to_tmp='./tmp_files', 
        weights_name='detector_weights_v2.pt',
        dataset_path='../dataset',
        submission_file_path='submission.csv',
        for_public = False
        ):

    model = Detector(path_to_weights=weights_path, path_to_tmp=path_to_tmp, weights_name=weights_name, confidence_level=0.1,box_intersetor_flag=True)
    
    image_paths = None

    if not for_public:
        image_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) if filename.endswith('.jpg')]
    else:
        image_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) if filename.endswith('.jpg') and filename.startswith('5 ')]

    predicts = []
    for img in tqdm(image_paths):
        predicts += model.predict([img])
        

    pd.DataFrame(format_output(predicts, dataset_path, for_public)).to_csv(submission_file_path, index=False, sep=';')

main(submission_file_path='weight_augmented.csv', for_public=True, weights_name='weights_augmented.pt')
main(submission_file_path='weight_augmented_last.csv', for_public=True, weights_name='weights_augmented_last.pt')
main(submission_file_path='V2_TRUE.csv', for_public=True, weights_name='detector_weights_v2.pt')