from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, Response
from models.Detector import Detector
from pathlib import Path
from PIL import Image
import aiofiles
import io
import os
import uvicorn

WORKDIR = ''

TMP_DIR = f'{WORKDIR}tmp_files/'
WEIGHTS_DIR = f'{WORKDIR}models/weights/'

app = FastAPI()

detector = Detector(path_to_weights=WEIGHTS_DIR, path_to_tmp=TMP_DIR, weights_name='detector_weights_v2.pt', confidence_level=0.1)

async def save_image(file_binary, filename= None):
    if filename is None:
        filename = f'tmp_{datetime.now()}_.jpg'

    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)

    img_file_path = os.path.join(TMP_DIR, filename)

    print(img_file_path)

    async with aiofiles.open(img_file_path, 'wb') as out_file:
        await out_file.write(file_binary) 

    return img_file_path, file_binary

@app.get("/")
def hello():
    return "Все робит"

@app.get("/get-image-by-path/")
async def get_image_by_path(img_path: str):
    image_path = Path(img_path)
    if not image_path.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(image_path)

@app.post("/detect/")
async def process_image(file: bytes = File(...)):
    if not file:
        return {"message": "No upload file sent"}
    else:
    
        img_file_path, binary_img_data = await save_image(file)

        response = detector.predict([img_file_path])[0]

        predict_img = Image.open(response['predict_img_path'])
        bytes_image = io.BytesIO()
        predict_img.save(bytes_image, format='PNG')

        names_count = response['names_count']
        formated_response = dict(list(zip(names_count.keys(), [str(i) for i in names_count.values()])))
        formated_response['predicted_image_path'] = response['predict_img_path']
        
        return Response(content=bytes_image.getvalue(), headers=formated_response, media_type="image/png")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=os.getenv('BACKEND_PORT'))
    # uvicorn.run(app, host="0.0.0.0", port=8228)
