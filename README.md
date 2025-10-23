# PILS-Trivision

## Objectif
Ce projet entraîne un modèle **YOLOv8** pour détecter et classifier différents types de déchets à partir du jeu de données **TACO (Trash Annotations in Context)**. 

L’objectif est de permettre la reconnaissance automatique d’objets jetés (plastique, métal, papier, etc.) afin de faire la classification automatiquement.

## Télécharger le dataset TACO
### 1. Ouvrir ton terminal

### 2. Cloner le dépôt officiel TACO
git clone https://github.com/pedropro/TACO.git

### 3. Entrer dans le dossier
cd TACO

### 4. Installer les dépendances requises
pip install -r requirements.txt

### 5. Télécharger les images et les annotations
python3 download.py

### 6. Convertir le dataset
exécuter la commande dessous:

cd ..

python .\taco_to_yolo_frbins_unified.py `
  --ann ".\TACO\data\annotations.json" `
  --images-root ".\TACO\data" `
  --out ".\datasets\taco_yolo_fr" `
  --val-ratio 0.2 `
  --copy-mode copy `
  --verbose

## Entraîner le modèle YOLOv8
### 1. Installation de YOLOv8
pip install ultralytics

### 2.Lancer l’entraînement
exécuter 'train.py'

## Tester le modèle entraîné
exécuter 'test.py'
