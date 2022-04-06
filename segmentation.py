import mmseg
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

import streamlit as st

config_file = '/home/aiteam/tykim/mmde/mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'

checkpoint_file = '/home/aiteam/tykim/mmde/mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

def load_model():
  model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
  return model


# @st.cache
def inference(model, image_tensor):
  # image = utils.load_image(image_filename)
  # transform = transforms.Compose([
  #   transforms.ToTensor(),
  #   transforms.Lambda(lambda x: x.mul(255))
  # ])
  # image = transform(image).unsqueeze(0).to(device)

  st.write(type(image_tensor))
  
  result = inference_segmentor(model, image_tensor)  
  return result