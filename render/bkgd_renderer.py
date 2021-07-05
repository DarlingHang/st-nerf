from abc import ABC, abstractmethod
from typing import Tuple, Optional
from collections import namedtuple

import numpy as np
from PIL import Image
import pyrender as pr

# skew is generally not supported
Pinhole = namedtuple('Pinhole', ['fx', 'fy', 'cx', 'cy'])


class MeshRender(ABC):
    @abstractmethod
    def load_mesh(self, fn: str) -> None:
        pass

    @abstractmethod
    def render(self, pinhole: Optional[Pinhole] = None,
               pose: Optional[np.ndarray] = None) -> Image.Image:
        pass


class PrRender(MeshRender):
    _gl_cv = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ])

    def __init__(self, resolution: Tuple[int, int]):
        self._scene = pr.Scene(ambient_light=np.ones(3))
        self._mesh = None
        self._cam = None

        self._width, self._height = resolution
        self._render = pr.OffscreenRenderer(self._width, self._height)

    def load_mesh(self, fn: str) -> None:
        tm = pr.mesh.trimesh.load_mesh(fn)
        mesh = pr.Mesh.from_trimesh(tm)
        if mesh:
            if self._mesh is not None:
                self._scene.remove_node(self._mesh)
            self._scene.add(mesh)
            self._mesh = mesh

    def render(self, pinhole: Optional[Pinhole] = None,
               pose: Optional[np.ndarray] = None,
               znear=pr.constants.DEFAULT_Z_NEAR,
               zfar=pr.constants.DEFAULT_Z_FAR) -> Image.Image:
        if pinhole is not None:
            # update intrinsics
            if self._cam is None:
                cam = pr.IntrinsicsCamera(*pinhole, znear, zfar)
                self._cam = self._scene.add(cam)
            else:
                cam = self._cam.camera
                cam.fx, cam.fy, cam.cx, cam.cy = pinhole
                cam.znear, cam.zfar = znear, zfar
        if self._cam is None:
            raise ValueError('Empty intrinsics while previous camera not set')

        # camera is not None
        if pose is not None:
            # update camera pose
            # gl_to_world = cv_to_world @ gl_to_cv
            self._scene.set_pose(self._cam, pose @ self._gl_cv)

        color, _ = self._render.render(self._scene)
        return color