from models.Detector import Detector
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, Response
from PIL import Image
import aiofiles
import io
import os
import uvicorn

WORKDIR = ''

TMP_DIR = f'{WORKDIR}tmp_files/'
WEIGHTS_DIR = f'{WORKDIR}models/weights/'

app = FastAPI()

detector = Detector(path_to_weights=WEIGHTS_DIR, path_to_tmp=TMP_DIR, weights_name='detector_weights_v2.pt')

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


@app.post("/detect/")
async def process_image(file: bytes = File(...)):
    if not file:
        return {"message": "No upload file sent"}
    else:
    
        img_file_path, binary_img_data = await save_image(file)

        # classifier_probs, recognited_text, predict_img_path = pipeline.forward(img_file_path)

        # pipeline = Pipeline(WEIGHTS_DIR, TMP_DIR, classifier_weights_name = 'weights_71_0.94_0.93_0.93_0.93.pt')

        # response = format_response_detect_client_prod(classifier_probs, recognited_text, predict_img_path)

        response = detector.predict([img_file_path])

        predict_img = Image.open(response['predict_img_path'])
        bytes_image = io.BytesIO()

        predict_img.save(bytes_image, format='PNG')
        
        return Response(content=bytes_image.getvalue(), headers={}, media_type="image/png")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=os.getenv('BACKEND_PORT'))
    # uvicorn.run(app, host="0.0.0.0", port=8228)
