# ST-NeRF in PyTorch

We provide PyTorch implementations for our paper:
[Editable Free-viewpoint Video Using a Layered Neural Representation](https://arxiv.org/abs/2104.14786)

SIGGRAPH 2021

Jiakai Zhang, Xinhang Liu, Xinyi Ye, Fuqiang Zhao, Yanshun Zhang, Minye Wu, Yingliang Zhang, Lan Xu and Jingyi Yu

**ST-NeRF: [Project](https://frankzhang0309.github.io/st-nerf/) |  [Paper](https://arxiv.org/abs/2104.14786)**

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/DarlingHang/ST-NeRF
cd ST-NeRF
```

- Install [PyTorch](http://pytorch.org) and other dependencies. You can create a new Conda environment using: 
```
conda env create -f environment.yml
```

### Datasets
The walking and taekwondo datasets can be downloaded from [here](https://drive.google.com/drive/folders/13YHw_YSGewvcgYdwqbelM9L2JiPNWLi7?usp=sharing).

### Apply a pre-trained model to render demo videos
- We provide our pretrained models which can be found under the `outputs` folder.
- We provide some example scripts under the `demo` folder.
- To run our demo scripts, you need to first downloaded the corresponding dataset, and put them under the folder specified by `DATASETS` -> `TRAIN` in `configs/config_taekwondo.yml` and `configs/config_walking.yml`
- For the walking sequence, you can render videos where some performers are hided by typing the command:
```
python demo/taekwondo_demo.py -c configs/config_taekwondo.yml
```
- For the taekwondo sequence, you can render videos where performers are translated and scaled by typing the command:
```
python demo/walking_demo.py -c configs/config_walking.yml
```
- The rendered images and videos will be under `outputs/taekwondo/rendered` and `outputs/walking/rendered`


## Citation
If you use this code for your research, please cite our papers.
```
@article{zhang2021editable,
  title={Editable Free-viewpoint Video Using a Layered Neural Representation},
  author={Zhang, Jiakai and Liu, Xinhang and Ye, Xinyi and Zhao, Fuqiang and Zhang, Yanshun and Wu, Minye and Zhang, Yingliang and Xu, Lan and Yu, Jingyi},
  journal={arXiv preprint arXiv:2104.14786},
  year={2021}
}
```
