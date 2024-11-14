import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Configura il dispositivo per l'elaborazione (GPU se disponibile)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il modello VGG16 pre-addestrato e mettilo in modalità eval
model = models.vgg16(pretrained=True).to(device)
model.eval()

# Trasformazioni per pre-elaborare le immagini come richiesto da VGG16
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Ridimensiona a 224x224 pixel
    transforms.ToTensor(),              # Converti in tensore
    transforms.Normalize(               # Normalizza usando i valori medi e std di ImageNet
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

# Percorso del dataset all'interno del repository
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')

# Carica le immagini dalla cartella del dataset e applica le trasformazioni
dataset = datasets.ImageFolder(dataset_path, transform=transform)

# Crea un DataLoader per caricare le immagini in batch
batch_size = 16  # Scegli la dimensione del batch
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Passa le immagini alla rete VGG16 in batch
with torch.no_grad():  # Disabilita il calcolo del gradiente
    for images, labels in data_loader:
        images = images.to(device)  # Sposta le immagini sul dispositivo selezionato
        outputs = model(images)     # Ottieni l'output della rete per il batch

        # outputs contiene le feature delle immagini, puoi processarlo o visualizzarlo
        print(outputs.shape)  # Stampa la forma dell'output per verificare il batch
