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
</div>

## Abstract
Content-preserving style transfer—generating stylized outputs based on content and style references—remains a significant challenge for Diffusion Transformers (DiTs) due to the inherent entanglement of content and style features in their internal representations. In this technical report, we present TeleStyle, a lightweight yet effective model for both image and video stylization. Built upon Qwen-Image-Edit, TeleStyle leverages the base model’s robust capabilities in content preservation and style customization. To facilitate effective training, we curated a high-quality dataset of distinct specific styles and further synthesized triplets using thousands of diverse, in-the-wild style categories. We introduce a Curriculum Continual Learning framework to train TeleStyle on this hybrid dataset of clean (curated) and noisy (synthetic) triplets. This approach enables the model to generalize to unseen styles without compromising precise content fidelity. Additionally, we introduce a video-to-video stylization module to enhance temporal consistency and visual quality. TeleStyle achieves state-of-the-art performance across three core evaluation metrics: style similarity, content consistency, and aesthetic quality.

## Latest News

- Jan 28, 2026: We release the <a href="https://github.com/Tele-AI/TeleStyle" target="_blank">code</a> and <a href="https://huggingface.co/Tele-AI/TeleStyle" target="_blank">model</a> of TeleStyle.
- - Jan 29, 2026: We release the <a href="[https://github.com/Tele-AI/TeleStyle](http://arxiv.org/abs/2601.20175)" target="_blank">technical report </a>  of TeleStyle.

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
- PyTorch 2.4.1 + CUDA 12.1
- diffusers 0.36.0
- transformers 4.49.0

### 2. Download Checkpoint

Download the [TeleStyle checkpoint](https://huggingface.co/Tele-AI/TeleStyle/tree/main) to a local path for example `weights/`:

We provide Image and Video checkpoint:

- **Image (reference style image + content image -> stylized image)**  
  diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors; diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors 37
  

- **Video (stylized first frame + content video -> stylized video)**  
  dit.ckpt; prompt_embeds.pth

### 3. Inference

We provide inference scripts for running TeleStyle on demo inputs for each task:

#### Image Stylization
```
python telestyleimage_inference.py --image_path assets/example/0.png --style_path videos/1.png --output_path results/image.png
```

#### Video Stylization
```
python telestylevideo_inference.py --video_path assets/example/1.mp4 --style_path assets/example/1-0.png --output_path results/video.mp4
```


## Citation
If you find TeleStyle useful in your research, please kindly cite our paper:
```bibtex
@article{teleai2026telestyle,
    title={TeleStyle: Content-Preserving Style Transfer in Images and Videos}, 
    author={Shiwen Zhang and Xiaoyan Yang and Bojia Zi and Haibin Huang and Chi Zhang and Xuelong Li},
    journal={arXiv preprint arXiv:2601.20175},
    year={2026}
}

