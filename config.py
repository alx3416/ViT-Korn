# dataset options
DATA_DIR = 'data/NTU'
OUTPUT_DIR = 'out'
INPUT_SIZE = 512
SPLITS = ['train', 'val']

# Training options
# Available models:
# googlenet, mobilenet_v3_large, resnet50, swin_v2_b, vit_b_16, vit_b_32
MODEL = ["googlenet", "mobilenet_v3_large", "resnet50", "swin_v2_b", "vit_b_16", "vit_b_32"]
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 25
