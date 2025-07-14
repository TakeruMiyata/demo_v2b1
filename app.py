import os
import zipfile
import numpy as np
from huggingface_hub import hf_hub_download
import streamlit as st
from PIL import Image
from torchvision.transforms import ToTensor
from skimage.metrics import peak_signal_noise_ratio as psnr

# ページ設定：最大幅で表示
st.set_page_config(layout="wide")

# ----------------------------
# データのダウンロード＆解凍
# ----------------------------
DATA_DIR = "data/data"

if not os.path.exists(DATA_DIR):
    with st.spinner("Downloading dataset..."):
        zip_path = hf_hub_download(
            repo_id="TakeruMiyata/Contrel_Fusion",  # ← あなたの Dataset 名
            filename="data.zip",
            token=st.secrets["hf_token"],           # ← secrets.toml に登録した token
            repo_type="dataset"
        )
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")  # data/ フォルダが含まれている前提
        st.success("Data extracted.")

# ----------------------------
# PSNR 計算関数
# ----------------------------
def compute_psnr(pred_path, gt_path):
    pred = ToTensor()(Image.open(pred_path).convert("L")).numpy()
    gt = ToTensor()(Image.open(gt_path).convert("L")).numpy()
    return psnr(gt, pred)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Image Comparison App")

# ファイル一覧を取得
image_list = sorted(os.listdir(os.path.join(DATA_DIR, "input")))
image_file = st.selectbox("Select Image", image_list)

if image_file:
    # 各画像読み込み
    input_img = Image.open(os.path.join(DATA_DIR, "input", image_file))
    pred_img = Image.open(os.path.join(DATA_DIR, "output", image_file))
    gt_img = Image.open(os.path.join(DATA_DIR, "gt", image_file))

    # 横3列表示
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(input_img, caption="Input", use_container_width=True)
        score = compute_psnr(os.path.join(DATA_DIR, "input", image_file),
                             os.path.join(DATA_DIR, "gt", image_file))
        st.markdown(f"**PSNR: {score:.2f}**")
    with col2:
        st.image(pred_img, caption="Output", use_container_width=True)
        score = compute_psnr(os.path.join(DATA_DIR, "output", image_file),
                             os.path.join(DATA_DIR, "gt", image_file))
        st.markdown(f"**PSNR: {score:.2f}**")
    with col3:
        st.image(gt_img, caption="Ground Truth", use_container_width=True)
