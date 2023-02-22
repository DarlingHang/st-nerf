# st-nerf

We provide PyTorch implementations for our paper:
[Editable Free-viewpoint Video Using a Layered Neural Representation](https://arxiv.org/abs/2104.14786)

SIGGRAPH 2021

Jiakai Zhang, Xinhang Liu, Xinyi Ye, Fuqiang Zhao, Yanshun Zhang, Minye Wu, Yingliang Zhang, Lan Xu and Jingyi Yu


<img src="images/teaser.jpg" width="800"/>

**st-nerf: [Project](https://jiakai-zhang.github.io/st-nerf/) |  [Paper](https://arxiv.org/abs/2104.14786)**

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/DarlingHang/st-nerf
cd st-nerf
```

- Install [PyTorch](http://pytorch.org) and other dependencies using: 
```
conda create -n st-nerf python=3.8.5
conda activate st-nerf    
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install imageio matplotlib
pip install yacs kornia robpy
```


### Datasets
The walking and taekwondo datasets can be downloaded from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/xliufe_connect_ust_hk/EjqArjZxmmtDplj_IrwlUq0BMUyG69zr5YqXFBxgku4rRQ?e=n2fSBs).

### Apply a pre-trained model to render demo videos
- We provide our pretrained models which can be found under the `outputs` folder.
- We provide some example scripts under the `demo` folder.
- To run our demo scripts, you need to first downloaded the corresponding dataset, and put them under the folder specified by `DATASETS` -> `TRAIN` in `configs/config_taekwondo.yml` and `configs/config_walking.yml`
- For the walking sequence, you can render videos where some performers are hided by typing the command:
```
python demo/walking_demo.py -c configs/config_walking.yml
```
- For the taekwondo sequence, you can render videos where performers are translated and scaled by typing the command:
```
python demo/taekwondo_demo.py -c configs/config_taekwondo.yml
```
- The rendered images and videos will be under `outputs/taekwondo/rendered` and `outputs/walking/rendered`

## Acknowlegements
We borrowed some codes from [Multi-view Neural Human Rendering (NHR)](https://github.com/wuminye/NHR).

## Citation
If you use this code for your research, please cite our papers.
```
@article{zhang2021editable,
  title={Editable free-viewpoint video using a layered neural representation},
  author={Zhang, Jiakai and Liu, Xinhang and Ye, Xinyi and Zhao, Fuqiang and Zhang, Yanshun and Wu, Minye and Zhang, Yingliang and Xu, Lan and Yu, Jingyi},
  journal={ACM Transactions on Graphics (TOG)},
  volume={40},
  number={4},
  pages={1--18},
  year={2021},
  publisher={ACM New York, NY, USA}
}
```
