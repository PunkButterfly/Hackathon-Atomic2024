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
            formated_predicts['rel_x'].append(round(parsed_coords[i][0], 6))
            formated_predicts['rel_y'].append(round(parsed_coords[i][1], 6))
            formated_predicts['width'].append(round(parsed_coords[i][2], 6))
            formated_predicts['height'].append(round(parsed_coords[i][3], 6))

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
        dataset_path='../test_dataset',
        submission_file_path='submission.csv',
        for_public = False,
        confidence_level=0.25,
        box_intersetor_flag=False,
        iou_treshold=0.8
        ):

    model = Detector(path_to_weights=weights_path, path_to_tmp=path_to_tmp, weights_name=weights_name, confidence_level=confidence_level, box_intersetor_flag=box_intersetor_flag,iou_treshold=iou_treshold)
    
    image_paths = None

    if not for_public:
        image_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) if filename.endswith('.jpg')]
    else:
        image_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) if filename.endswith('.jpg') and filename.startswith('5 ')]

    predicts = []
    for img in tqdm(image_paths):
        predicts += model.predict([img])
        

    pd.DataFrame(format_output(predicts, dataset_path, for_public)).to_csv(submission_file_path, index=False, sep=';')

# main(submission_file_path='weights_augmented.csv', for_public=True, weights_name='weights_augmented.pt')  0.879
# main(submission_file_path='weights_augmented_last.csv', for_public=True, weights_name='weights_augmented_last.pt') 0.884
# main(submission_file_path='detector_weights_v2.csv', for_public=True, weights_name='detector_weights_v2.pt') 0.906

# main(submission_file_path='weights_augmented_cf_005_iou_06.csv', for_public=True, weights_name='weights_augmented.pt', confidence_level=0.05,box_intersetor_flag=True, iou_treshold=0.6) 0.849
# main(submission_file_path='weights_augmented_last_cf_005_iou_06.csv', for_public=True, weights_name='weights_augmented_last.pt', confidence_level=0.05,box_intersetor_flag=True, iou_treshold=0.6) 0.855
# main(submission_file_path='detector_weights_v2_cf_005_iou_06.csv', for_public=True, weights_name='detector_weights_v2.pt', confidence_level=0.05,box_intersetor_flag=True, iou_treshold=0.6) 0.86

# main(submission_file_path='weights_augmented_v2_cf_015_iou_075.csv', for_public=True, weights_name='weights_augmented.pt', confidence_level=0.15,box_intersetor_flag=True, iou_treshold=0.75) 0.869
# main(submission_file_path='weights_augmented_last_v2_cf_015_iou_075.csv', for_public=True, weights_name='weights_augmented_last.pt', confidence_level=0.15,box_intersetor_flag=True, iou_treshold=0.75)
# main(submission_file_path='detector_weights_v2_v2_cf_015_iou_075.csv', for_public=True, weights_name='detector_weights_v2.pt', confidence_level=0.15,box_intersetor_flag=True, iou_treshold=0.75) 0.889


# main(submission_file_path='detector_weights_v2_cf_015_iou_03.csv', for_public=True, weights_name='detector_weights_v2.pt', confidence_level=0.15,box_intersetor_flag=True, iou_treshold=0.3) 0.888
# main(submission_file_path='detector_weights_v2_cf_025_iou_02.csv', for_public=True, weights_name='detector_weights_v2.pt', confidence_level=0.25,box_intersetor_flag=True, iou_treshold=0.2) 0.911
# main(submission_file_path='detector_weights_v2_cf_001_iou_01.csv', for_public=True, weights_name='detector_weights_v2.pt', confidence_level=0.01,box_intersetor_flag=True, iou_treshold=0.1) 0.838

# main(submission_file_path='detector_weights_v2_cf_025_iou_005.csv', for_public=True, weights_name='detector_weights_v2.pt', confidence_level=0.25,box_intersetor_flag=True, iou_treshold=0.05) 0.911
# main(submission_file_path='detector_weights_v2_cf_04.csv', for_public=True, weights_name='detector_weights_v2.pt', confidence_level=0.4) 0.926
# main(submission_file_path='detector_weights_v2_cf_018_iou_04.csv', for_public=True, weights_name='detector_weights_v2.pt', confidence_level=0.18,box_intersetor_flag=True, iou_treshold=0.4) 0.898

# main(submission_file_path='weights_augmented_cf_005_iou_06.csv', weights_name='weights_augmented.pt', confidence_level=0.05,box_intersetor_flag=True, iou_treshold=0.6)  # 0.849

# main(submission_file_path='detector_weights_v2_cf_025_iou_005.csv', weights_name='detector_weights_v2.pt', confidence_level=0.25,box_intersetor_flag=True, iou_treshold=0.05)#  0.911
# main(submission_file_path='detector_weights_v2_cf_025_iou_02.csv', weights_name='detector_weights_v2.pt', confidence_level=0.25,box_intersetor_flag=True, iou_treshold=0.2) # 0.911
# main(submission_file_path='detector_weights_v2.csv', weights_name='detector_weights_v2.pt')#  0.906

# main(submission_file_path='epoch6_cf_025_iou_005.csv', weights_name='epoch6.pt', confidence_level=0.25,box_intersetor_flag=True, iou_treshold=0.05) # 0.911
# main(submission_file_path='epoch6_cf_025_iou_02.csv', weights_name='epoch6.pt', confidence_level=0.25,box_intersetor_flag=True, iou_treshold=0.2) #  0.911
# main(submission_file_path='epoch6.csv', weights_name='epoch6.pt') # 0.906

main(submission_file_path='epoch6_05.csv', weights_name='epoch6.pt', confidence_level=0.5) # 0.906
main(submission_file_path='epoch6_07.csv', weights_name='epoch6.pt', confidence_level=0.7) # 0.906
main(submission_file_path='epoch6_08.csv', weights_name='epoch6.pt', confidence_level=0.8) # 0.906

main(submission_file_path='detector_weights_v2_05.csv', weights_name='detector_weights_v2.pt', confidence_level=0.5) # 0.906
main(submission_file_path='detector_weights_v2_07.csv', weights_name='detector_weights_v2.pt', confidence_level=0.7) # 0.906
main(submission_file_path='detector_weights_v2_08.csv', weights_name='detector_weights_v2.pt', confidence_level=0.8) # 0.906

main(submission_file_path='detector_weights_v2_05_06.csv', weights_name='detector_weights_v2.pt', confidence_level=0.5, iou_treshold= 0.6) # 0.906
main(submission_file_path='detector_weights_v2_07_06.csv', weights_name='detector_weights_v2.pt', confidence_level=0.7, iou_treshold= 0.6) # 0.906
main(submission_file_path='detector_weights_v2_08_06.csv', weights_name='detector_weights_v2.pt', confidence_level=0.8, iou_treshold= 0.6) # 0.906

main(submission_file_path='weights_augmented_05_06.csv', weights_name='weights_augmented.pt', confidence_level=0.5, iou_treshold= 0.6) # 0.906
main(submission_file_path='weights_augmented_07_06.csv', weights_name='weights_augmented.pt', confidence_level=0.7, iou_treshold= 0.6) # 0.906
main(submission_file_path='weights_augmented_08_06.csv', weights_name='weights_augmented.pt', confidence_level=0.8, iou_treshold= 0.6) # 0.906