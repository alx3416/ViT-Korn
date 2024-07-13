import torch
from torchvision import models
import torch.nn as nn
import time
import os
from tempfile import TemporaryDirectory
import config as ini


def set_model():
    if ini.MODEL == "googlenet":
        model = models.googlenet(weights='IMAGENET1K_V1')
    elif ini.MODEL == "resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
    elif ini.MODEL == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
    elif ini.MODEL == "swin_v2_b":
        model = models.swin_v2_b(weights='IMAGENET1K_V1')
    elif ini.MODEL == "vit_b_16":
        model = models.vit_b_16(weights='IMAGENET1K_V1')
    elif ini.MODEL == "vit_b_32":
        model = models.vit_b_32(weights='IMAGENET1K_V1')
    return model


def set_transfer(model_ft, class_names):
    if ini.MODEL == "googlenet" or ini.MODEL == "resnet50":
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    elif ini.MODEL == "swin_v2_b":
        num_ftrs = model_ft.head.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    elif ini.MODEL == "vit_b_16" or ini.MODEL == "vit_b_32":
        num_ftrs = model_ft.heads.head.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    elif ini.MODEL == "mobilenet_v3_large":
        num_ftrs = model_ft.classifier[0].in_features
        model_ft.classifier = nn.Sequential(
            nn.Linear(in_features=num_ftrs, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
        )

    return model_ft


def set_criterion_loss():
    return nn.CrossEntropyLoss()


def set_train_model(model, criterion, optimizer, scheduler, dataloaders, device,
                    dataset_sizes, num_epochs=ini.EPOCHS):
    since = time.time()

    # output directory to save training checkpoints
    best_model_params_path = os.path.join(ini.OUTPUT_DIR, 'best_checkpoint_' + ini.MODEL + '.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model
