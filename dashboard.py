# streamlit
from select import select
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components
# DL
import torch
import torchvision
import numpy as np
import pandas as pd
# import cv2

# visulaization
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
# utils
from pathlib import Path
import requests
# my module
import classification as cls
import utils

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="TY's Lab", page_icon='😁',
                   layout="wide")

st.title("MY Lab")

menu_opt = ('Main', 'MNIST', 'FFCV',)
selected_menu = st.sidebar.selectbox('Select Menu', menu_opt)


if selected_menu == 'MNIST':
  
  st.markdown("<h1 align='center'; >MNIST Classification</h1>", unsafe_allow_html=True)
  st.write('MNIST 모델을 통한 연습 페이지')
  st.write('모델에 들어갈 이미지를 선택하거나 직접 그리기')

  img_radio_method = st.radio( "Choose", ('select', 'draw'))

  col1, col2, col3 = st.columns([1,4,1])
  
  
  if img_radio_method == 'select':
    img_num = st.selectbox('Select Image', ('0.jpg', '1.jpg', '2.jpg',
                                        '3.jpg', '4.jpg', '5.jpg','6.jpg',
                                        '7.jpg','8.jpg', '9.jpg'))

    img_path= "./data/mnist/" + img_num
    st.write("### Source Image:")
    image = Image.open(img_path)
    col2.image(image, width=400, use_column_width=True)

    # transform into torch tensor
    img_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    st.write(f'{img_tensor.shape}')
      
  elif img_radio_method == 'draw':
    st.write('draw box')
  
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    bg_color = st.color_picker("Background color hex: ", "#eee")
    # bg_image = col1.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = st.checkbox("Update in realtime", True)


    canvas_result = st_canvas(
      fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
      stroke_width=stroke_width,
      stroke_color='#FFFFFF',
      background_color='#000000',
      # background_image=Image.open(bg_image) if bg_image else None,
      update_streamlit=realtime_update,
      height=200,
      width=200,
      drawing_mode="freedraw",
      key="canvas",)


    if canvas_result.image_data is not None:
      img_tensor = utils.canvas_to_tensor(canvas_result)
      # img_tensor = torchvision.transforms.ToTensor()(input_numpy_array).unsqueeze(0)
      st.write(f'{img_tensor.shape}')
        
  clicked = st.button("Classification")

  if clicked:
  
    with st.spinner('Loading the model...'):
      model_name = 'mnist'
      model = cls.load_model(model_name)
      # st.write('')  
    st.success('Loading the model.. Done!')
    st.balloons()
    device = 'cpu'
    preds = cls.inference(model, img_tensor)
    # st.write(f'{preds.shape}')
    # st.write(f'{preds}')
    # st.write(f'{np.exp(preds)}')
    df = pd.DataFrame(np.vstack((preds, np.exp(preds))).T, columns=['logits', 'probability'])
    st.table(df)


    prob_array = np.exp(preds).T
    df = pd.DataFrame(prob_array, columns=['prob'], index=['0','1','2',
                                                       '3', '4','5','6','7','8','9'])
                                                       
    fig = px.bar(df,log_y=True)
    st.plotly_chart(fig, use_container_width=True)


if selected_menu == 'FFCV':
  st.markdown("<h1 align='center';> FFCV </h1>", unsafe_allow_html=True)

  model_name = st.selectbox('Select Model', ('EfficientNet', 'Reset'))
