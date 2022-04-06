import streamlit as st
import utils
import torch
from torchvision import transforms
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dict = {'mnist' : BasicModel(),}

@st.cache
def load_model(model_name):
  with torch.no_grad():
    model = model_dict[model_name]
    model_path = './saved_models/' + model_name + '.pth'
    # state_dict = torch.load(model_path)
    # model.load_state_dict(state_dict)
    model = torch.load(model_path)
    # model.to(device)
    model.eval()
    return model


@st.cache
def inference(model, image_tensor):
  # image = utils.load_image(image_filename)
  # transform = transforms.Compose([
  #   transforms.ToTensor(),
  #   transforms.Lambda(lambda x: x.mul(255))
  # ])
  # image = transform(image).unsqueeze(0).to(device)

  with torch.no_grad():
    output = model(image_tensor)

  return output.squeeze(0).detach().cpu().numpy()