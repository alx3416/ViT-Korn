import utils as ut
import config as ini
import preprocessing as pre
import modelling as mdll
import torch
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing

model_name = ini.MODEL
# model = mdll.set_model(model_name)
# model.load_state_dict(torch.load("out/" + model_name + "/" + model_name + '.pt'))
# model.eval()

# Definir transformaciones para preprocesar las imágenes
transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),  # Ajustar tamaño a lo que el modelo espera
    transforms.ToTensor(),  # Convertir a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar
])
model_name = model_name[0]
# Definir la ruta del modelo entrenado y la carpeta base de imágenes de prueba
modelo_ruta = "out/" + model_name + "/" + model_name + ".pt"
carpeta_base_imagenes = 'data/corn/val/'


def get_confusion_matrix(diabetes_y_test, diabetes_y_pred, target_names):
    cnf_matrix = confusion_matrix(diabetes_y_test, diabetes_y_pred)
    print(classification_report(diabetes_y_test, diabetes_y_pred, target_names=target_names))
    return cnf_matrix


# Definir una clase Dataset personalizada para cargar las imágenes
class ImagenesDataset(Dataset):
    def __init__(self, carpeta_base, transform=None):
        self.carpeta_base = carpeta_base
        self.transform = transform

        # Obtener la lista de clases (nombre de las carpetas)
        self.lista_clases = os.listdir(self.carpeta_base)
        self.imagenes = []
        self.labels = []

        # Iterar sobre cada clase
        for i, clase in enumerate(self.lista_clases):
            carpeta_clase = os.path.join(self.carpeta_base, clase)
            lista_imagenes = os.listdir(carpeta_clase)
            for imagen in lista_imagenes:
                self.imagenes.append(os.path.join(carpeta_clase, imagen))
                self.labels.append(i)

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, idx):
        imagen_path = self.imagenes[idx]
        imagen = Image.open(imagen_path)

        if self.transform:
            imagen = self.transform(imagen)

        label = self.labels[idx]

        return imagen, label


# Crear una instancia del Dataset
dataset = ImagenesDataset(carpeta_base_imagenes, transformaciones)

# Crear un DataLoader para manejar los lotes de datos
batch_size = 4
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
device = pre.set_device()

# Cargar el modelo
model = mdll.set_model(model_name)
modelo = torch.load(modelo_ruta)
modelo.eval()
modelo.to(device)

y_pred = []
y_test = []
# Iterar sobre los lotes de datos
for imagenes_batch, etiquetas_batch in data_loader:
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

names = ['Gray_leaf_spot', 'Common_rust', 'healthy', 'Northern_leaf_blight']
confusion_matrix = get_confusion_matrix(y_pred, y_test, names)

