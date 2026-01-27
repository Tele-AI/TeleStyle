
import os
import cv2
import json
import time
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from typing import List, Dict, Optional, Tuple
from einops import rearrange
from omegaconf import OmegaConf
from decord import VideoReader
from diffusers.utils import export_to_video
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from telestylevideo_transformer import WanTransformer3DModel
from telestylevideo_pipeline import WanPipeline


def load_video(video_path: str, video_length: int) -> torch.Tensor:
    if "png" in video_path.lower() or "jpeg" in video_path.lower() or "jpg" in video_path.lower():
        image = cv2.imread(video_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        image = image[None, None]  # 添加 batch 和 frame 维度
        image = torch.from_numpy(image) / 127.5 - 1.0
        return image
    
    vr = VideoReader(video_path)
    frames = list(range(min(len(vr), video_length)))
    images = vr.get_batch(frames).asnumpy()
    images = torch.from_numpy(images) / 127.5 - 1.0
    images = images[None]  # 添加 batch 维度
    return images

class VideoStyleInference:
    """
    视频风格转换推理类
    """
    def __init__(self, config: Dict):
        """
        初始化推理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device(f"cuda:0")
        self.random_seed = config['random_seed']
        self.video_length = config['video_length']
        self.H = config['height']
        self.W = config['width']
        self.num_inference_steps = config['num_inference_steps']
        self.vae_path = os.path.join(config['ckpt_t2v_path'], "vae")
        self.transformer_config_path = os.path.join(config['ckpt_t2v_path'], "transformer_config.json")
        self.scheduler_path = os.path.join(config['ckpt_t2v_path'], "scheduler")
        self.ckpt_path = config['ckpt_dit_path']
        self.output_path = config['output_path']
        self.prompt_embeds_path = config['prompt_embeds_path']
        
        # 加载模型
        self._load_models()
    
    def _load_models(self):
        """
        加载模型和权重
        """
        # 加载状态字典
        state_dict = torch.load(self.ckpt_path, map_location="cpu")["transformer_state_dict"]
        transformer_state_dict = {}
        for key in state_dict:
            transformer_state_dict[key.split("module.")[1]] = state_dict[key]
        
        # 加载配置
        config = OmegaConf.to_container(
            OmegaConf.load(self.transformer_config_path)
        )
        
        # 初始化模型
        self.vae = AutoencoderKLWan.from_pretrained(self.vae_path, torch_dtype=torch.float16).to(self.device)
        self.transformer = WanTransformer3DModel(**config)
        self.transformer.load_state_dict(transformer_state_dict)
        self.transformer = self.transformer.to(self.device).half()
        self.scheduler = UniPCMultistepScheduler.from_pretrained(self.scheduler_path)
        
        # 初始化管道
        self.pipe = WanPipeline(
            transformer=self.transformer, 
            vae=self.vae, 
            scheduler=self.scheduler
        )
        self.pipe.to(self.device)
    
    def inference(self, source_videos: torch.Tensor, first_images: torch.Tensor, video_path: str, step: int) -> torch.Tensor:
        """
        执行风格转换推理
        
        Args:
            source_videos: 源视频张量
            first_images: 风格参考图像张量
            video_path: 源视频路径
            step: 推理步骤索引
            
        Returns:
            生成的视频张量
        """
        source_videos = source_videos.to(self.device).half()
        first_images = first_images.to(self.device).half()
        prompt_embeds_ = torch.load(self.prompt_embeds_path).to(self.device).half()

        print(f"Source videos shape: {source_videos.shape}, First images shape: {first_images.shape}")
        
        latents_mean = torch.tensor(self.vae.config.latents_mean)
        latents_mean = latents_mean.view(1, 16, 1, 1, 1).to(self.device, torch.float16)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std)
        latents_std = latents_std.view(1, 16, 1, 1, 1).to(self.device, torch.float16)

        bsz = 1
        _, _, h, w, _ = source_videos.shape
        
        if h < w:
            output_h, output_w = self.H, self.W
        else:
            output_h, output_w = self.W, self.H

        with torch.no_grad():
            # 处理源视频
            source_videos = rearrange(source_videos, "b f h w c -> (b f) c h w")
            source_videos = F.interpolate(source_videos, (output_h, output_w), mode="bilinear")
            source_videos = rearrange(source_videos, "(b f) c h w -> b c f h w", b=bsz)

            # 处理风格参考图像
            first_images = rearrange(first_images, "b f h w c -> (b f) c h w")
            first_images = F.interpolate(first_images, (output_h, output_w), mode="bilinear")
            first_images = rearrange(first_images, "(b f) c h w -> b c f h w", b=bsz)

            # 编码到潜在空间
            source_latents = self.vae.encode(source_videos).latent_dist.mode()
            source_latents = (source_latents - latents_mean) * latents_std

            first_latents = self.vae.encode(first_images).latent_dist.mode()
            first_latents = (first_latents - latents_mean) * latents_std

            neg_first_latents = self.vae.encode(torch.zeros_like(first_images)).latent_dist.mode()
            neg_first_latents = (neg_first_latents - latents_mean) * latents_std

        video = self.pipe(
            source_latents=source_latents,
            first_latents=first_latents,
            neg_first_latents=neg_first_latents,
            num_frames=self.video_length,
            guidance_scale=3.0,
            height=output_h,
            width=output_w,
            prompt_embeds_=prompt_embeds_,
            num_inference_steps=self.num_inference_steps,
            generator=torch.Generator(device=self.device).manual_seed(self.random_seed),
        ).frames[0]
        
        return video

if __name__ == "__main__":
    config = {
        "random_seed": 42,
        "video_length": 129,
        "height": 720,
        "width": 1248,
        "num_inference_steps": 25,
        "ckpt_t2v_path": "./Wan2.1-T2V-1.3B-Diffusers",
        "ckpt_dit_path": "weights/dit.ckpt",
        "prompt_embeds_path": "weights/prompt_embeds.pth",
        "output_path": "./results"
    }
    
    # 初始化推理器
    inference_engine = VideoStyleInference(config)
    
    data_list = [
        {
            "video_path": "assets/example/2.mp4",
            "image_path": "assets/example/2-0.png"
        },
        {
            "video_path": "assets/example/2.mp4",
            "image_path": "assets/example/2-1.png"
        },
    ]

    for step, data in enumerate(data_list):
        video_path = data['video_path']
        style_image_path = data['image_path']
        
        source_video = load_video(video_path, config['video_length'])
        style_image = Image.open(style_image_path)
        style_image = np.array(style_image)
        style_image = torch.from_numpy(style_image) / 127.5 - 1.0
        style_image = style_image[None, None, :, :, :]  # 添加 batch 和 frame 维度
        
        with torch.no_grad():
            generated_video = inference_engine.inference(source_video, style_image, video_path, step)
        
        os.makedirs(config['output_path'], exist_ok=True)
        output_filename = f"{config['output_path']}/{step}.mp4"
        export_to_video(generated_video, output_filename)
                
