import torch
from torchvision import models
import torch.nn as nn
import time
import os
import config as ini
import utils as ut


def set_model(model_name):
    if model_name == "googlenet":
        model = models.googlenet(weights='IMAGENET1K_V1')
    elif model_name == "resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
    elif model_name == "swin_v2_b":
        model = models.swin_v2_b(weights='IMAGENET1K_V1')
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights='IMAGENET1K_V1')
    elif model_name == "vit_b_32":
        model = models.vit_b_32(weights='IMAGENET1K_V1')
    return model


def set_transfer(model_ft, class_names, model_name):
    if model_name == "googlenet" or model_name == "resnet50":
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    elif model_name == "swin_v2_b":
        num_ftrs = model_ft.head.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    elif model_name == "vit_b_16" or model_name == "vit_b_32":
        num_ftrs = model_ft.heads.head.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    elif model_name == "mobilenet_v3_large":
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
                    dataset_sizes, model_name, num_epochs=ini.EPOCHS):
    since = time.time()

    # output directory to save training checkpoints
    ut.check_output_folder("out/" + model_name)
    best_model_params_path = os.path.join("out/" + model_name + "/", 'best_checkpoint_' + model_name + '.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0
    history_train_loss = []
    history_train_acc = []
    history_val_loss = []
    history_val_acc = []
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
            if phase == 'train':
                history_train_loss.append(epoch_loss)
                history_train_acc.append(epoch_acc.detach().item())
            elif phase == 'val':
                history_val_loss.append(epoch_loss)
                history_val_acc.append(epoch_acc.detach().item())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model, history_train_loss, history_train_acc, history_val_loss, history_val_acc


def save_historic_data(train_loss, train_accuracy, val_loss, val_accuracy, model_name):
    ut.check_output_folder("out/" + model_name)
    with open("out/" + model_name + "/training_loss_" + model_name + ".txt", "w") as output:
        output.write(str(train_loss))
    with open("out/" + model_name + "/training_accuracy_" + model_name + ".txt", "w") as output:
        output.write(str(train_accuracy))
    with open("out/" + model_name + "/val_loss_" + model_name + ".txt", "w") as output:
        output.write(str(val_loss))
    with open("out/" + model_name + "/val_accuracy_" + model_name + ".txt", "w") as output:
        output.write(str(val_accuracy))
