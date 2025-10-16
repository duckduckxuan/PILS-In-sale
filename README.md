# PILS-In-sale
# Détection de déchets avec YOLOv8 et le jeu de données TACO

## Objectif
Ce projet entraîne un modèle **YOLOv8** pour détecter et classifier différents types de déchets à partir du jeu de données **TACO (Trash Annotations in Context)**. 

L’objectif est de permettre la reconnaissance automatique d’objets jetés (plastique, métal, papier, etc.) afin de faire la classification automatiquement.

## Télécharger le dataset TACO
### ① Ouvrir ton terminal

### ② Cloner le dépôt officiel TACO
git clone https://github.com/pedropro/TACO.git

### ③ Entrer dans le dossier
cd TACO

### ④ Installer les dépendances requises
pip install -r requirements.txt

### ⑤ Télécharger les images et les annotations
python3 download.py

### ⑥ Convertir le dataset
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
### ① Installation de YOLOv8
pip install ultralytics

### ② Lancer l’entraînement
exécuter 'train.py'

## Tester le modèle entraîné
exécuter 'test.py'
