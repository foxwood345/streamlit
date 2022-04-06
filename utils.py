import torch
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt

import streamlit as st

def load_image(filename, size=None, scale=None):
  img = Image.open(filename).convert('RGB')
  if size is not None:
    img = img.resize((size, size), Image.ANTIALIAS)

  elif scale is not None:
    img = img.resize((int(img.size[0] / scale), int(img.size(1)/ scale)),
                     Image.ANTIALIAS)

  return img


def save_image(filename, data):
  img = data.clone().clamp(0, 255).numpy()
  img = img.transpose(1,2,0).astype('unit8')
  img = Image.fromarray(img)
  img.save(filename)


def get_pichart(ndarray, labels):
  """
  utils.get_pichart(np.exp(preds), ['0', '1','2','3','4',
                                      '5','6','7','8','9'])
  """
  fig = plt.figure(figsize=(10, 4))
  plt.pie(ndarray, labels = labels)
  st.pyplot(fig)



def canvas_to_tensor(canvas):
  """
  Convert Image of RGBA to single channel B/W and convert from numpy array
  to a PyTorch Tensor of [1,1,28,28]
  """
  img = canvas.image_data
  img = img[:, :, :-1]  # Ignore alpha channel
  img = img.mean(axis=2)
  img = img/255
  img = img*2 - 1.
  img = torch.FloatTensor(img)
  tens = img.unsqueeze(0).unsqueeze(0)
  tens = F.interpolate(tens, (28, 28))
  return tens