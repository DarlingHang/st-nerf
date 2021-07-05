from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.deep_rgb = True

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.COARSE_RAY_SAMPLING = 64
_C.MODEL.FINE_RAY_SAMPLING = 80
_C.MODEL.SAMPLE_METHOD = "NEAR_FAR"
_C.MODEL.BOARDER_WEIGHT = 1e10
_C.MODEL.SAME_SPACENET = False

_C.MODEL.TKERNEL_INC_RAW = True
_C.MODEL.POSE_REFINEMENT = True
_C.MODEL.USE_DIR = True
_C.MODEL.REMOVE_OUTLIERS = False
_C.MODEL.TRAIN_BY_POINTCLOUD = False
_C.MODEL.USE_DEFORM_VIEW = False # Use deformnet to deform view inconsisdency
_C.MODEL.USE_DEFORM_TIME = False # Use deformnet to deform time inconsisdency
_C.MODEL.BKGD_USE_DEFORM_TIME = False
_C.MODEL.BKGD_USE_SPACE_TIME = False
_C.MODEL.USE_SPACE_TIME = False
_C.MODEL.DEEP_RGB = True




# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [400,250]
# Size of the image during test
_C.INPUT.SIZE_TEST = [400,250]
# Size of the image during sample layer
_C.INPUT.SIZE_LAYER = [400,250]
# Minimum scale for the image during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.1307, ]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.3081, ]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ""
_C.DATASETS.TMP_RAYS = "rays_tmp"
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.SHIFT = 0.0
_C.DATASETS.MAXRATION = 0.0
_C.DATASETS.ROTATION = 0.0
_C.DATASETS.USE_MASK = False
_C.DATASETS.NUM_FRAME = 1
_C.DATASETS.FACTOR = 1
_C.DATASETS.FIXED_NEAR = -1.0
_C.DATASETS.FIXED_FAR = -1.0

_C.DATASETS.CENTER_X = 0.0
_C.DATASETS.CENTER_Y = 0.0
_C.DATASETS.CENTER_Z = 0.0
_C.DATASETS.SCALE = 1.0
_C.DATASETS.FILE_OFFSET = 0
_C.DATASETS.FRAME_OFFSET = 0
_C.DATASETS.FRAME_NUM = 0
_C.DATASETS.LAYER_NUM = 0
_C.DATASETS.CAMERA_NUM = 0
_C.DATASETS.BKGD_SAMPLE_RATE = 0.1
_C.DATASETS.CAMERA_STEPSIZE = 1

_C.DATASETS.USE_LABEL = False
_C.DATASETS.VIEW_MASK = None
_C.DATASETS.FIXED_LAYER = []

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.BUNCH = 4096
_C.SOLVER.START_ITERS=50
_C.SOLVER.END_ITERS=200
_C.SOLVER.LR_SCALE=0.1
_C.SOLVER.COARSE_STAGE = 10

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

_C.SOLVER.BBOX_ID = 0

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
