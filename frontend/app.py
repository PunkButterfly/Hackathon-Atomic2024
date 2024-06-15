from datetime import datetime
import os
import requests as rq
import streamlit as st

# URL = 'http://158.160.17.229:8520'
URL = f"http://backend:{os.getenv('BACKEND_PORT')}"
# URL = 'http://0.0.0.0:8228'

TMP_DIR  = './tmp_files'

MANUAL = """ \
### Расшифровка типов дефектов
- ```прилегающие дефекты``` - брызги, прожоги от дуги, название класса ```adj```
- ```дефекты целостности``` - кратер, шлак, свищ, пора, прожог, включения, название класса - ```int```
- ```дефекты геометрии```   - подрез, непровар, наплыв, чешуйчатость, западание, неравномерность, название класса - ```geo```
- ```дефекты постобработки```     - заусенец, торец, задир, забоина, название класса - ```pro```
- ```дефекты невыполнения```          - незаполнение раковины, несплавление, название класса - ```non```\
"""

def parse_detected_classes(headers):
    curr_cls = {}
    classes = ["adj", "int", "geo", "pro", "non"]
    for cls in classes:
        if cls in headers:
            curr_cls[cls] = headers[cls]

    return curr_cls

def save_bin_image(image_bin):
    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)
    img_path = f"{TMP_DIR}/tmp_{datetime.now()}.jpg"
    with open (img_path, 'wb') as f:
        f.write(image_bin)

    return img_path

st.set_page_config(page_title="ATOMIC HACK 2024", layout="wide")

st.title("Punk Butterfly - Atomic Hack 2024", anchor=False)
st.header("")

uploaded_file = st.file_uploader('Фото сварочного шва', accept_multiple_files=False)

response = None

if uploaded_file:
    source_img = uploaded_file.read()
    response = rq.post(f"{URL}/detect/", files={"file": source_img})

if response:
    image_bin = response.content
    headers = response.headers

    curr_image_path = save_bin_image(image_bin)

image, response_body = st.columns(2, gap='small')

with image:
    if uploaded_file:
        st.image(curr_image_path, caption="Выделенные дефекты на снимке")

with response_body:
    if uploaded_file:
        result_string = ""

        curr_classes = parse_detected_classes(headers)
        if len(curr_classes.keys()) > 0:
            result_string += "### Обнаружено:\n\n"
            for cls, val in curr_classes.items():
                result_string += f"* :red[{val}] дефектов типа **{cls}**\n\n"

            result_string += MANUAL
        else:
            result_string += "Дефектов не обнаружено."

        st.markdown(result_string)