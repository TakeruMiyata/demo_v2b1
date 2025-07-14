import streamlit as st
from PIL import Image
import os
import numpy as np
from torchvision.transforms import ToTensor
from skimage.metrics import peak_signal_noise_ratio as psnr

st.set_page_config(layout="wide")

def compute_psnr(pred_path, gt_path):
    pred = ToTensor()(Image.open(pred_path).convert("L")).numpy()
    gt = ToTensor()(Image.open(gt_path).convert("L")).numpy()
    return psnr(gt, pred)

# UI
st.title("Image Comparison (v2b1)")

image_list = os.listdir("data/input")
image_file = st.selectbox("Select Image", image_list)

if image_file:
    input_path = os.path.join("data/input", image_file)
    pred_path = os.path.join("data/output", image_file)
    gt_path = os.path.join("data/gt", image_file)

    input_img = Image.open(input_path)
    pred_img = Image.open(pred_path)
    gt_img = Image.open(gt_path)

    input_psnr = compute_psnr(input_path, gt_path)
    output_psnr = compute_psnr(pred_path, gt_path)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(input_img, caption="Input", use_container_width=True)
        st.markdown(f"**Input PSNR:** {input_psnr:.2f}")

    with col2:
        st.image(pred_img, caption="Output", use_container_width=True)
        st.markdown(f"**Output PSNR:** {output_psnr:.2f}")

    with col3:
        st.image(gt_img, caption="Ground Truth", use_container_width=True)
