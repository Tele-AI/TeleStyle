
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import gc
import subprocess
import cv2
import json
import time
import torch
import argparse
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

import atexit
import signal

def _cleanup_gpu():
    """Release GPU memory on exit or interrupt."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _kill_stale_gpu_processes():
    """Kill any other processes using the GPU (leftover from previous interrupted runs)."""
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            pid = int(parts[0].strip())
            mem_mib = int(parts[1].strip())
            if pid != my_pid and mem_mib > 100:
                print(f"Killing stale GPU process {pid} (using {mem_mib} MiB VRAM)")
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
        # Wait for GPU memory to be freed
        time.sleep(2)
    except Exception as e:
        print(f"Warning: Could not check for stale GPU processes: {e}")

atexit.register(_cleanup_gpu)
signal.signal(signal.SIGINT, lambda *_: (_cleanup_gpu(), exit(1)))
signal.signal(signal.SIGTERM, lambda *_: (_cleanup_gpu(), exit(1)))


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
        # Clear any leftover GPU memory from previous interrupted runs
        gc.collect()
        torch.cuda.empty_cache()
        self.random_seed = config['random_seed']
        self.video_length = config['video_length']
        self.H = config['height']
        self.W = config['width']
        self.num_inference_steps = config['num_inference_steps']
        self.vae_path = os.path.join(config['ckpt_t2v_path'], "vae")
        self.transformer_config_path = os.path.join(config['ckpt_t2v_path'], "transformer", "config.json")
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
        self.vae.enable_tiling()
        # Keep transformer on CPU initially to avoid OOM when both models are on GPU
        self.transformer = WanTransformer3DModel(**config)
        self.transformer.load_state_dict(transformer_state_dict)
        self.transformer = self.transformer.half()  # fp16 on CPU
        self.scheduler = UniPCMultistepScheduler.from_pretrained(self.scheduler_path)
        
        # 初始化管道 (transformer stays on CPU until inference)
        self.pipe = WanPipeline(
            transformer=self.transformer, 
            vae=self.vae, 
            scheduler=self.scheduler
        )
    
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

            # Offload transformer to CPU to free VRAM for VAE encoding
            self.transformer.to("cpu")
            torch.cuda.empty_cache()

            # 编码到潜在空间
            source_latents = self.vae.encode(source_videos).latent_dist.mode()
            source_latents = (source_latents - latents_mean) * latents_std

            first_latents = self.vae.encode(first_images).latent_dist.mode()
            first_latents = (first_latents - latents_mean) * latents_std

            neg_first_latents = self.vae.encode(torch.zeros_like(first_images)).latent_dist.mode()
            neg_first_latents = (neg_first_latents - latents_mean) * latents_std

            # Move VAE to CPU and transformer to GPU for diffusion
            self.vae.to("cpu")
            self.transformer.to(self.device)
            torch.cuda.empty_cache()

        # Run pipeline, returning latents to avoid VAE decode while transformer is on GPU
        result = self.pipe(
            source_latents=source_latents,
            first_latents=first_latents,
            neg_first_latents=neg_first_latents,
            num_frames=self.video_length,
            guidance_scale=1.0,
            height=output_h,
            width=output_w,
            prompt_embeds_=prompt_embeds_,
            num_inference_steps=self.num_inference_steps,
            generator=torch.Generator(device=self.device).manual_seed(self.random_seed),
            output_type="latent",
        )
        latents = result.frames  # raw latent tensor when output_type="latent"

        # Offload transformer to CPU, bring VAE back to GPU for decode
        self.transformer.to("cpu")
        torch.cuda.empty_cache()
        self.vae.to(self.device)

        latents = latents.to(self.vae.dtype)
        latents_mean_dec = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std_dec = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std_dec + latents_mean_dec

        with torch.no_grad():
            video = self.vae.decode(latents, return_dict=False)[0]
        video = self.pipe.video_processor.postprocess_video(video, output_type="pil")[0]
        # Convert PIL frames to numpy for export_to_video compatibility
        video = [np.array(frame) for frame in video]

        return video

def parse_args():
    parser = argparse.ArgumentParser(description='视频风格转换推理')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    parser.add_argument('--video_length', type=int, default=129, help='视频长度')
    parser.add_argument('--height', type=int, default=720, help='输出高度')
    parser.add_argument('--width', type=int, default=1248, help='输出宽度')
    parser.add_argument('--num_inference_steps', type=int, default=25, help='推理步数')
    parser.add_argument('--ckpt_t2v_path', type=str, default="./Wan2.1-T2V-1.3B-Diffusers", help='T2V检查点路径')
    parser.add_argument('--ckpt_dit_path', type=str, default="weights/dit.ckpt", help='DiT检查点路径')
    parser.add_argument('--prompt_embeds_path', type=str, default="weights/prompt_embeds.pth", help='提示嵌入路径')
    parser.add_argument('--output_path', type=str, default="./results_video", help='输出路径')
    parser.add_argument('--video_path', type=str, required=True, help='源视频路径')
    parser.add_argument('--image_path', type=str, required=True, help='风格参考图像路径')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Kill any stale GPU processes from previous interrupted runs
    _kill_stale_gpu_processes()
    
    args = parse_args()
    
    config = {
        "random_seed": args.random_seed,
        "video_length": args.video_length,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "ckpt_t2v_path": args.ckpt_t2v_path,
        "ckpt_dit_path": args.ckpt_dit_path,
        "prompt_embeds_path": args.prompt_embeds_path,
        "output_path": args.output_path
    }
    
    # 初始化推理器
    inference_engine = VideoStyleInference(config)
    
    video_path = args.video_path
    style_image_path = args.image_path
    
    source_video = load_video(video_path, config['video_length'])
    style_image = Image.open(style_image_path)
    style_image = np.array(style_image)
    style_image = torch.from_numpy(style_image) / 127.5 - 1.0
    style_image = style_image[None, None, :, :, :]  # 添加 batch 和 frame 维度
    
    with torch.no_grad():
        generated_video = inference_engine.inference(source_video, style_image, video_path, 0)
    
    os.makedirs(config['output_path'], exist_ok=True)
    output_filename = f"{config['output_path']}/generated_video.mp4"
    export_to_video(generated_video, output_filename)
                
