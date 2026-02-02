# <div align="center">TeleStyle:  Content-Preserving Style Transfer in Images and Videos</div>
<div align="center">
    Shiwen Zhang, Xiaoyan Yang, Bojia Zi, Haibin Huang, Chi Zhang, Xuelong Li
    <br>
    Institute of Artificial Intelligence, China Telecom (TeleAI) 
</div>
<br>
<div align="center">
    [<a href="https://tele-ai.github.io/TeleStyle/" target="_blank">Project Page</a>]
    [<a href="http://arxiv.org/abs/2601.20175" target="_blank">arXiv</a>]
    [<a href="https://huggingface.co/Tele-AI/TeleStyle" target="_blank">Hugging Face</a>]
    [<a href="https://github.com/Tele-AI/TeleStyle" target="_blank">GitHub</a>]
    [<a href="https://huggingface.co/spaces/witcherderivia/TeleStyle" target="_blank">Demo</a>]
</div>

## Abstract
Content-preserving style transfer—generating stylized outputs based on content and style references—remains a significant challenge for Diffusion Transformers (DiTs) due to the inherent entanglement of content and style features in their internal representations. In this technical report, we present TeleStyle, a lightweight yet effective model for both image and video stylization. Built upon Qwen-Image-Edit, TeleStyle leverages the base model’s robust capabilities in content preservation and style customization. To facilitate effective training, we curated a high-quality dataset of distinct specific styles and further synthesized triplets using thousands of diverse, in-the-wild style categories. We introduce a Curriculum Continual Learning framework to train TeleStyle on this hybrid dataset of clean (curated) and noisy (synthetic) triplets. This approach enables the model to generalize to unseen styles without compromising precise content fidelity. Additionally, we introduce a video-to-video stylization module to enhance temporal consistency and visual quality. TeleStyle achieves state-of-the-art performance across three core evaluation metrics: style similarity, content consistency, and aesthetic quality.

## Latest News
- Jan 30, 2026: We refine the code and update requirements.txt. In addition, a new version of TeleStyle-Image model with better performance has been uploaded. Finally, we release a [free online demo for TeleStyle-Image ](https://huggingface.co/spaces/witcherderivia/TeleStyle). Please light a star to support this project if you find the demo useful. 
- Jan 28, 2026: We release the <a href="http://arxiv.org/abs/2601.20175" target="_blank">technical report </a>, <a href="https://github.com/Tele-AI/TeleStyle" target="_blank">code</a> and <a href="https://huggingface.co/Tele-AI/TeleStyle" target="_blank">model</a> of TeleStyle.

## Todo List

- [x] Release inference code
- [x] Release models
- [x] Release technical report



## How to use

### 1. Installation

```
pip install -r requirements.txt
```

This environment is tested with:
- Python 3.11
- PyTorch 2.9.1 + CUDA 12.1
- diffusers 0.36.0
- transformers 4.57.3

### 2. Download Checkpoint

Download the [TeleStyle checkpoint](https://huggingface.co/Tele-AI/TeleStyle/tree/main) to a local path for example `weights/`:

We provide Image and Video checkpoint:

- **Image (reference style image + content image -> stylized image)**  
  diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors; diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors
  

- **Video (stylized first frame + content video -> stylized video)**  
  dit.ckpt; prompt_embeds.pth

### 3. Inference

We provide inference scripts for running TeleStyle-Image and TeleStyle-Video:

#### Image Stylization
```
python telestyleimage_inference.py
```

#### Video Stylization
```
python telestylevideo_inference.py --video_path assets/example/1.mp4 --image_path assets/example/1-0.png --output_path results
```


## Citation
If you find TeleStyle useful in your research, please light a star for the project and cite our paper, thank you:
```bibtex
@article{teleai2026telestyle,
    title={TeleStyle: Content-Preserving Style Transfer in Images and Videos}, 
    author={Shiwen Zhang and Xiaoyan Yang and Bojia Zi and Haibin Huang and Chi Zhang and Xuelong Li},
    journal={arXiv preprint arXiv:2601.20175},
    year={2026}
}

