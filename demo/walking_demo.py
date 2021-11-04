import argparse
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
from os import mkdir
import shutil
import torch
import torch.nn.functional as F
import random
from torchvision import utils as vutils
import numpy as np
import imageio
import matplotlib.pyplot as plt

sys.path.append('..')
from config import cfg
from engine.layered_trainer import do_train, do_evaluate
#from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR,build_scheduler
from layers import make_loss
from utils.logger import setup_logger
from layers.RaySamplePoint import RaySamplePoint
from utils import batchify_ray, vis_density
from render import LayeredNeuralRenderer

text = 'This is the program to render the nerf by the specific frame id and layer id, try to get help by using '
parser = argparse.ArgumentParser(description=text)
parser.add_argument('-c', '--config', default='', help='set the config file path to render the network')
parser.add_argument('-g','--gpu', type=int, default=0, help='set gpu id to render the network')
args = parser.parse_args()

torch.cuda.set_device(args.gpu)
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)


cfg.merge_from_file(args.config)
cfg.freeze()

for i in range(16):
    neural_renderer = LayeredNeuralRenderer(cfg)

    density_threshold = 20 # Can be set to higher to hide glass
    bkgd_density_threshold = 0.8 
    inverse_y_axis = False # For some y-inversed model

    neural_renderer.set_fps(25)
    #neural_renderer.set_near(0.1)

    neural_renderer.set_save_count(i)
    neural_renderer.set_pose_duration(i,15) # [ min , max )
    neural_renderer.set_smooth_path_poses(75, around=False)
    neural_renderer.invert_poses()

    #do_evaluate(neural_renderer.model, neural_renderer.dataset)

    neural_renderer.set_save_dir("origin")
    neural_renderer.render_path(inverse_y_axis,density_threshold,bkgd_density_threshold,auto_save=True)
    # neural_renderer.save_video()

'''
neural_renderer.hide_layer(1)
neural_renderer.set_save_dir("hide_man_1")
neural_renderer.render_path(inverse_y_axis,density_threshold,bkgd_density_threshold,auto_save=True)
neural_renderer.save_video()


neural_renderer.hide_layer(2)
neural_renderer.set_save_dir("hide_both")
neural_renderer.render_path(inverse_y_axis,density_threshold,bkgd_density_threshold,auto_save=True)
neural_renderer.save_video()
'''