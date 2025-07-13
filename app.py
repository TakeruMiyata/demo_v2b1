import gradio as gr
from PIL import Image
from torchvision.transforms import ToTensor
from skimage.metrics import peak_signal_noise_ratio
import os

def compute_psnr(pred_path, gt_path):
    pred = ToTensor()(Image.open(pred_path).convert("L")).numpy()
    gt = ToTensor()(Image.open(gt_path).convert("L")).numpy()
    return peak_signal_noise_ratio(gt, pred, data_range=1.0)

def compare_image(file_name):
    pred_path = f"data/output/{file_name}"
    gt_path = f"data/gt/{file_name}"
    input_path = f"data/input/{file_name}"
    
    psnr = compute_psnr(pred_path, gt_path)
    
    return (
        Image.open(input_path),
        Image.open(pred_path),
        Image.open(gt_path),
        f"PSNR: {psnr:.2f} dB"
    )

demo = gr.Interface(
    fn=compare_image,
    inputs=gr.Dropdown(choices=os.listdir("data/input"), label="Select Image"),
    outputs=[
        gr.Image(label="Input"),
        gr.Image(label="Output"),
        gr.Image(label="Ground Truth"),
        gr.Textbox(label="PSNR")
    ],
    title="Image Comparison & PSNR Viewer"
)

if __name__ == "__main__":
    demo.launch()
