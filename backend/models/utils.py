import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

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

# Определение классов и соответствующих им цветов
class_colors = {
    "adj": "red",
    "int": "green",
    "geo": "blue",
    "pro": "yellow",
    "non": "purple"
}

def draw_bounding_boxes(src_image_path, dst_image_path, names, coords):
    """
    Отображает bounding boxes и подписывает их именами классов на изображении.

    :param image_path: Путь к исходному изображению
    :param names: Список имен классов
    :param coords: Список относительных координат bounding boxes в формате xyxy
    """
    # Открываем изображение
    img = Image.open(src_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # try:

    file = open("./arial.ttf", "rb")
    bytes_font = BytesIO(file.read())
    font = ImageFont.truetype(bytes_font,size=56)
    # except IOError:
        # font = ImageFont.load_default()

    for name, coord in zip(names, coords):
        # Преобразование относительных координат в абсолютные
        x1, y1, x2, y2 = coord
        
        # Рисование bounding box
        draw.rectangle([x1, y1, x2, y2], outline=class_colors[name], width=16)
        
        # Добавление подписи класса рядом с bounding box
        text_size = draw.textsize(name, font)
        # Подложка для текста для лучшего выделения подписи
        draw.text((x1, y1 - text_size[1]), name, fill="white",font=font)
    
    # Отображение изображения с bounding boxes
    img.save(dst_image_path)

    return dst_image_path