# Balanced Rate–Distortion Optimization in Learned Image Compression

PyTorch implementation for  
**Balanced Rate–Distortion Optimization in Learned Image Compression (CVPR 2025)**  

📄 Paper: [Balanced Rate–Distortion Optimization in Learned Image Compression (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Balanced_Rate-Distortion_Optimization_in_Learned_Image_Compression_CVPR_2025_paper.html)

---

## Requirements

- Python 3.9+  
- PyTorch (CUDA recommended)  
- compressai  
- timm  
- torch-ema  
- tqdm  
- accelerate  

Install dependencies:
```bash
pip install torch torchvision    # pick CUDA version as needed
pip install compressai timm torch-ema tqdm accelerate
```

## Example usage for elic
### Standard Training
```
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py --cuda   --lambda 0.013 --epochs 150  
```
Mixed precision training:
```
CUDA_VISIBLE_DEVICES=0  accelerate launch --mixed_precision fp16 train.py --cuda   --lambda 0.013 --epochs 150  
```
Use bf16 if fp16 is unstable.

### Balanced Training

```
CUDA_VISIBLE_DEVICES=0 accelerate launch train_balanced.py --cuda --lambda 0.013 --epochs 150  --gamma 0.003
```
gamma needs to be tuned for each model.

## Citation
If you find this repository useful, please cite:
```
@inproceedings{zhang2025balanced,
  title={Balanced rate-distortion optimization in learned image compression},
  author={Zhang, Yichi and Duan, Zhihao and Huang, Yuning and Zhu, Fengqing},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={2428--2438},
  year={2025}
}
```