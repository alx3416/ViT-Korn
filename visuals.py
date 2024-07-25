import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import utils as ut
import seaborn as sns


def imshow(inputs, title=None):
    """Display image for Tensor."""
    out = torchvision.utils.make_grid(inputs)
    out = out.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    out = std * out + mean
    out = np.clip(out, 0, 1)
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, device, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def save_plot_losses(train_loss, val_loss, model_name):
    ut.check_output_folder("out/" + model_name)
    plt.figure()
    plt.plot(train_loss, label='training')
    plt.plot(val_loss, label='validation')
    plt.title(model_name + ' model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("out/" + model_name + "/" + model_name + '_loss.svg', format='svg')


def save_plot_accuracies(train_acc, val_acc, model_name):
    ut.check_output_folder("out/" + model_name)
    plt.figure()
    plt.plot(train_acc, label='training')
    plt.plot(val_acc, label='validation')
    plt.title(model_name + ' model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig("out/" + model_name + "/" + model_name + '_accuracy.svg', format='svg')


def save_confusion_matrix(confusion, model_name, class_names):
    new_fig = plt.figure()
    sns.heatmap(confusion, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.savefig("out/" + model_name + "/" + model_name + '_confusion_matrix.svg', format='svg')
    plt.close(new_fig)
