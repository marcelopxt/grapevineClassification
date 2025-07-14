import streamlit as st
import gdown as gd
import tensorflow as tf

def load_model():
    #https://drive.google.com/file/d/1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp/view?usp=sharing
    url = 'https://drive.google.com/uc?id=1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp'
    gd.down(url, 'modelo_quantizado16bits.tflite')
    interpreter = tf.lite.Interpreter(model_path="modelo_quantizado16bits.tflite")
    interpreter.alocate_tensors()
    return interpreter


def main():
    st.set_page_config(page_title="Classiica folhas de Videiras!")
    st.write("# Classifica Folhas de Videiras!")
    
    
if __name__ == "__main__":
    main()


