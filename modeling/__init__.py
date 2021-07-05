# encoding: utf-8

from .layered_rfrender import LayeredRFRender

def build_layered_model(cfg,camera_num=0,scale=None,shift=None):
    model = LayeredRFRender(cfg, camera_num=camera_num, scale=scale,shift=shift)
    return model
