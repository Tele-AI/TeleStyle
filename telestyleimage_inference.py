import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

import os

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

telestyle_image= hf_hub_download(repo_id="Tele-AI/TeleStyle", filename="weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors")

speedup = hf_hub_download(repo_id="Tele-AI/TeleStyle", filename="weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")

pipe.load_lora(pipe.dit,telestyle_image)
pipe.load_lora(pipe.dit,speedup)



content_ref='' #content reference image
style_ref=''#style reference image
    
prompt = 'Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.'




w,h=Image.open(content_ref).convert("RGB").size



minedge=1024
if w>h:
    r=w/h
    h=minedge
    w=int(h*r)-int(h*r)%16
    
else:
    r=h/w
    w=minedge
    h=int(w*r)-int(w*r)%16

images = [
    Image.open(content_ref).convert("RGB").resize((w, h)),
    Image.open(style_ref).convert("RGB").resize((minedge, minedge)) ,
]



image = pipe(prompt, edit_image=images, seed=123, num_inference_steps=4, height=h, width=w,edit_image_auto_resize=False,cfg_scale=1.0)#ligtning



save_dir=f'./telestyle_image_output/'

os.makedirs(save_dir,exist_ok=True)
prefix=style_ref.split('/')[-1].split('.')[0]


image.save(os.path.join(save_dir, f'{prefix}_result.png'))


print(f"saved to {os.path.join(save_dir, f'{prefix}_result.png')}")
        
