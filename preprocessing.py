from torchvision import datasets, transforms
import config as ini
import torch
import os


def augmentation_normalization():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(ini.INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(ini.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(ini.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


def load_image_dataset(data_transforms):
    image_datasets = {x: datasets.ImageFolder(os.path.join(ini.DATA_DIR, x),
                                              data_transforms[x])
                      for x in ini.SPLITS}
    return image_datasets


def load_image_test_dataset(data_transforms):
    image_datasets = {x: datasets.ImageFolder(os.path.join(ini.DATA_DIR, x),
                                              data_transforms[x])
                      for x in ['test']}
    return image_datasets


def data_loading(image_datasets):
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=ini.BATCH_SIZE,
                                                  shuffle=True, num_workers=ini.NUM_WORKERS)
                   for x in ['train', 'val']}
    return dataloaders


def data_test_loading(image_datasets):
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=ini.BATCH_SIZE,
                                                  shuffle=False, num_workers=ini.NUM_WORKERS)
                   for x in ['test']}
    return dataloaders


def get_dataset_sizes(image_datasets):
    dataset_sizes = {x: len(image_datasets[x]) for x in ini.SPLITS}
    return dataset_sizes


def get_test_dataset_sizes(image_datasets):
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
    return dataset_sizes


def get_classes_names(image_datasets):
    class_names = image_datasets['train'].classes
    return class_names


def get_test_classes_names(image_datasets):
    class_names = image_datasets['test'].classes
    return class_names


def set_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_training_batch_sample(dataloaders):
    inputs, classes = next(iter(dataloaders['train']))
    return inputs, classes


def get_test_batch_sample(dataloaders):
    inputs, classes = next(iter(dataloaders['test']))
    return inputs, classes
