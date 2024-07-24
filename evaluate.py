import preprocessing as pre
import visuals as vis
import modelling as mdll
import config as ini
import utils as ut
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch

cudnn.benchmark = True
plt.ion()  # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = pre.augmentation_normalization()

image_datasets = pre.load_image_test_dataset(data_transforms)
dataloaders = pre.data_test_loading(image_datasets)
dataset_sizes = pre.get_test_dataset_sizes(image_datasets)
class_names = pre.get_test_classes_names(image_datasets)

device = pre.set_device()

inputs, classes = pre.get_test_batch_sample(dataloaders)
vis.imshow(inputs, title=[class_names[x] for x in classes])

print("ok")
