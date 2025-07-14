import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource
def carrega_modelo():
    #https://drive.google.com/file/d/1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp/view?usp=sharing
    url = 'https://drive.google.com/uc?id=1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp'

    # baixa o arquivo (modelo)
    gdown.download(url, 'modelo_quantizado16bits.tflite')

    # carrega o modelo
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')

    # disponiliza para uso
    interpreter.allocate_tensors()

    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader('Arraste e solte a uma imagem ou clique para selecionar uma' , type=['png','jpg','jpeg'] )

    if uploaded_file is not None:
        # ler a imagem
        image_data = uploaded_file.read()

        # abrir a imagem
        image = Image.open(io.BytesIO(image_data))

        # exibir a imagem na página
        st.image(image)
        st.success('Imagem foi carregada com sucesso!')

        # converter a imagem em ponto flutuante
        image = np.array(image, dtype=np.float32)

        # normalizar a imagem
        image = image / 255.0

        # adicionar uma dimensão extra
        image = np.expand_dims(image, axis=0)

        return image


def main():
    st.set_page_config(page_title="Classifica Folhas de Videiras!")

    st.write("# Classifica Folhas de Videiras!")

    # carregar o modelo
    interpreter = carrega_modelo()

    # carregar a imagem
    image = carrega_imagem()

    # classificar a imagem

if __name__ == "__main__":
    main()