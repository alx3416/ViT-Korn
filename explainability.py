from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import resnet50
import cv2
import numpy as np
import torch
from torchsummary import summary

model_name = "mobilenet_v3_large"
# model = resnet50(pretrained=True)
model_path = "out/" + model_name + "/" + model_name + ".pt"
model = torch.load(model_path)
model.eval()
print(model)
target_layers = [model.features[-1]]

rgb_img = cv2.imread("data/corn/test/01_Cercospora_leaf_spot/2174d811-b088-4b31-8088-1738630b94d4___RS_GLSp 4467.JPG", 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to("cpu")

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers)

targets = [ClassifierOutputTarget(0)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
cv2.imshow("GradCam", visualization)
cv2.imwrite("out/" + model_name + "/01_cercospora.png", visualization)
cv2.waitKey(0)
