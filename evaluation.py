import utils as ut
import config as ini
import preprocessing as pre
import modelling as mdll
import torch
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader

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
carpeta_base_imagenes = 'data/corn/test/'


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

# Cargar el modelo
model = mdll.set_model(model_name)
modelo = model.load_state_dict(torch.load(modelo_ruta))
modelo.eval()

# Iterar sobre los lotes de datos
for imagenes_batch, etiquetas_batch in data_loader:
    # Pasar el batch de imágenes a través del modelo
    with torch.no_grad():
        salidas = modelo(imagenes_batch)

    # Obtener las clases predichas para cada imagen en el batch
    _, predicciones = torch.max(salidas, 1)

    # Aquí puedes manejar las predicciones como lo necesites
    for i in range(batch_size):
        imagen_actual = imagenes_batch[i]
        etiqueta_actual = etiquetas_batch[i]
        prediccion_actual = predicciones[i]

        print(
            f'Predicción para imagen {i + 1} del batch: Clase {prediccion_actual.item()} (Real: {etiqueta_actual.item()})')
