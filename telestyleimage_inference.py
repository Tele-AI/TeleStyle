import torch
import argparse
import os
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from huggingface_hub import hf_hub_download


class ImageStyleInference:
    """
    图像风格转换推理类
    """
    def __init__(self, config: dict):
        """
        初始化推理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = config['random_seed']
        self.min_edge_size = config['min_edge_size']
        self.num_inference_steps = config['num_inference_steps']
        self.cfg_scale = config['cfg_scale']
        self.output_path = config['output_path']
        
        # 加载模型
        self._load_models()
    
    def _load_models(self):
        """
        加载模型和权重
        """
        self.pipe = QwenImagePipeline.from_pretrained(
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

        self.pipe.load_lora(self.pipe.dit, telestyle_image)
        self.pipe.load_lora(self.pipe.dit, speedup)

    def inference(self, content_ref: str, style_ref: str) -> Image.Image:
        """
        执行风格转换推理
        
        Args:
            content_ref: 内容参考图像路径
            style_ref: 风格参考图像路径
            
        Returns:
            生成的图像
        """
        prompt = 'Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.'

        w, h = Image.open(content_ref).convert("RGB").size

        minedge = self.min_edge_size
        if w > h:
            r = w / h
            h = minedge
            w = int(h * r) - int(h * r) % 16
        else:
            r = h / w
            w = minedge
            h = int(w * r) - int(w * r) % 16

        images = [
            Image.open(content_ref).convert("RGB").resize((w, h)),
            Image.open(style_ref).convert("RGB").resize((minedge, minedge)),
        ]

        image = self.pipe(
            prompt, 
            edit_image=images, 
            seed=self.random_seed, 
            num_inference_steps=self.num_inference_steps, 
            height=h, 
            width=w,
            edit_image_auto_resize=False,
            cfg_scale=self.cfg_scale
        )  # lightning

        return image


def parse_args():
    parser = argparse.ArgumentParser(description='图像风格转换推理')
    parser.add_argument('--random_seed', type=int, default=123, help='随机种子')
    parser.add_argument('--min_edge_size', type=int, default=1024, help='最小边尺寸')
    parser.add_argument('--num_inference_steps', type=int, default=4, help='推理步数')
    parser.add_argument('--cfg_scale', type=float, default=1.0, help='条件场引导缩放')
    parser.add_argument('--output_path', type=str, default='./results_image/', help='输出路径')
    parser.add_argument('--image_path', type=str, required=True, help='内容参考图像路径')
    parser.add_argument('--style_path', type=str, required=True, help='风格参考图像路径')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    config = {
        "random_seed": args.random_seed,
        "min_edge_size": args.min_edge_size,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "output_path": args.output_path
    }
    
    # 初始化推理器
    inference_engine = ImageStyleInference(config)
    
    image_path = args.image_path
    style_path = args.style_path
    
    with torch.no_grad():
        generated_image = inference_engine.inference(image_path, style_path)
    
    os.makedirs(config['output_path'], exist_ok=True)
    prefix = style_path.split('/')[-1].split('.')[0]
    output_filename = os.path.join(config['output_path'], f'{prefix}_result.png')
    
    generated_image.save(output_filename)
    print(f"saved to {output_filename}")
        
