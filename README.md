# PILS-In-sale
# Détection de déchets avec YOLOv8 et le jeu de données TACO

## Objectif
Ce projet entraîne un modèle **YOLOv8** pour détecter et classifier différents types de déchets à partir du jeu de données **TACO (Trash Annotations in Context)**. 

L’objectif est de permettre la reconnaissance automatique d’objets jetés (plastique, métal, papier, etc.) afin de faire la classification automatiquement.

## Télécharger le dataset TACO
### Ouvrir ton terminal

### Cloner le dépôt officiel TACO
git clone https://github.com/pedropro/TACO.git

### Entrer dans le dossier
cd TACO

### Installer les dépendances requises
pip install -r requirements.txt

### Télécharger les images et les annotations
python3 download.py

## Entraîner le modèle YOLOv8
### Installation de YOLOv8
pip install ultralytics

### Télécharger garbage.yaml

### Lancer l’entraînement
yolo detect train data=garbage.yaml model=yolov8n.pt epochs=50 imgsz=640 project=runs/detect name=best

## Tester le modèle entraîné
### Avec la caméra intégrée
yolo detect predict model=runs/detect/best/weights/best.pt source=0 show=True

### Avec une caméra USB
yolo detect predict model=runs/detect/best/weights/best.pt source=1 show=True
