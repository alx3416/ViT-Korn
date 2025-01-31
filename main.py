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

def train():
    cudnn.benchmark = True
    plt.ion()  # interactive mode

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = pre.augmentation_normalization()

    image_datasets = pre.load_image_dataset(data_transforms)
    dataloaders = pre.data_loading(image_datasets)
    dataset_sizes = pre.get_dataset_sizes(image_datasets)
    class_names = pre.get_classes_names(image_datasets)

    device = pre.set_device()

    inputs, classes = pre.get_training_batch_sample(dataloaders)
    vis.imshow(inputs, title=[class_names[x] for x in classes])

    for model_name in ini.MODEL:
        model_ft = mdll.set_model(model_name)
        model_ft = mdll.set_transfer(model_ft, class_names, model_name)
        model_ft = model_ft.to(device)
        criterion = mdll.set_criterion_loss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft, train_loss, train_acc, val_loss, val_acc = mdll.set_train_model(model_ft, criterion, optimizer_ft,
                                                                                  exp_lr_scheduler,
                                                                                  dataloaders, device, dataset_sizes,
                                                                                  model_name,
                                                                                  num_epochs=ini.EPOCHS)

        ut.check_output_folder("out/" + model_name)
        torch.save(model_ft.state_dict(), "out/" + model_name + "/" + model_name + '_state_dict.pt')
        torch.save(model_ft, "out/" + model_name + "/" + model_name + '.pt')
        mdll.save_historic_data(train_loss, train_acc, val_loss, val_acc, model_name)
        vis.save_plot_losses(train_loss, val_loss, model_name)
        vis.save_plot_accuracies(train_acc, val_acc, model_name)

if __name__ == '__main__':
    train()
