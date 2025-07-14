import os
import zipfile
import numpy as np
from huggingface_hub import hf_hub_download
import streamlit as st
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr


# ----------------------
# データダウンロード & 解凍
# ----------------------
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    with st.spinner("Downloading dataset..."):
        zip_path = hf_hub_download(
            repo_id="TakeruMiyata/Contrel_Fusion",  # あなたのDataset名に変更
            filename="data.zip",                   # zipファイル名
            token=st.secrets["hf_token"],          # secrets.toml に登録したトークン
            repo_type="dataset"
        )
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        st.success("Data extracted.")

# ----------------------
# Streamlit UI
# ----------------------
st.title("Image Comparison (v2b1)")

# 画像一覧取得
image_list = sorted(os.listdir(os.path.join(DATA_DIR, "input")))
image_name = st.selectbox("Select Image", image_list)

# 画像パス構築
input_path = os.path.join(DATA_DIR, "input", image_name)
output_path = os.path.join(DATA_DIR, "pred", image_name)
gt_path = os.path.join(DATA_DIR, "gt", image_name)

# 画像読み込み
input_img = Image.open(input_path)
output_img = Image.open(output_path)
gt_img = Image.open(gt_path)

# レイアウト：3列表示
col1, col2, col3 = st.columns(3)
with col1:
    st.image(input_img, caption="Input", use_column_width=True)
    st.write("PSNR: -")
with col2:
    st.image(output_img, caption="Output", use_column_width=True)
    score = psnr(np.array(gt_img), np.array(output_img))
    st.write(f"PSNR: {score:.2f}")
with col3:
    st.image(gt_img, caption="Ground Truth", use_column_width=True)
