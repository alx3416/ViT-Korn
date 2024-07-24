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
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_confusion_matrix(diabetes_y_test, diabetes_y_pred, target_names):
    cnf_matrix = confusion_matrix(diabetes_y_test, diabetes_y_pred)
    print(classification_report(diabetes_y_test, diabetes_y_pred, target_names=target_names))
    report = classification_report(diabetes_y_test, diabetes_y_pred, target_names=target_names, output_dict=True)
    print(report)
    pandas_report = pd.DataFrame(report).transpose()
    pandas_report.to_csv("out/classification_report.csv")
    return cnf_matrix


def save_confusion_matrix(confusion):
    new_fig = plt.figure()
    sns.heatmap(confusion, annot=True, cmap='Blues')
    plt.savefig("out/confusion_matrix.png")
    plt.close(new_fig)

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

model_name = ini.MODEL
model_name = model_name[0]

# Cargar el modelo
modelo_ruta = "out/" + model_name + "/" + model_name + ".pt"
model = mdll.set_model(model_name)
modelo = torch.load(modelo_ruta)
modelo.eval()
modelo.to(device)

y_pred = []
y_test = []
ut.check_output_folder("out/" + model_name)
# Iterar sobre los lotes de datos
for imagenes_batch, etiquetas_batch in dataloaders['test']:
    imagenes_batch = imagenes_batch.to(device)
    etquetas_batch = etiquetas_batch.to(device)
    # Pasar el batch de imágenes a través del modelo
    with torch.no_grad():
        salidas = modelo(imagenes_batch)

    # Obtener las clases predichas para cada imagen en el batch
    _, predicciones = torch.max(salidas, 1)

    # Aquí puedes manejar las predicciones como lo necesites
    for i in range(etiquetas_batch.shape[0]):
        imagen_actual = imagenes_batch[i]
        etiqueta_actual = etiquetas_batch[i]
        prediccion_actual = predicciones[i]
        y_pred.append(prediccion_actual.item())
        y_test.append(etiqueta_actual.item())
print("total de etiquetas: ", len(y_pred))

confusion_matrix = get_confusion_matrix(y_pred, y_test, class_names)
save_confusion_matrix(confusion_matrix)
