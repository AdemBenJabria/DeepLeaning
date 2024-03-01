import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
import time


# Enregistrement du temps de début
start_time = time.time()

# Labels des classes CIFAR-10
class_names = ['avion', 'voiture', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'bateau', 'camion']

# Chargement et préparation des données CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalisation
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Configuration de la data augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)

# Chargement du modèle pré-entraîné MobileNetV2 sans la partie classification
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Gel des poids du modèle pré-entraîné pour ne pas les modifier pendant l'entraînement
base_model.trainable = True

# Création du modèle en ajoutant des couches personnalisées
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle avec data augmentation
history = model.fit(data_augmentation.flow(x_train, y_train, batch_size=64), epochs=100, validation_data=(x_test, y_test))

# Calcul et affichage du temps d'entraînement total
end_time = time.time()
total_time = end_time - start_time
print(f"Temps d'entraînement total : {total_time:.2f} secondes")

# Prédiction des classes pour les données de test avec le modèle
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calcul et affichage du temps d'entraînement total
end_time = time.time()
total_time = end_time - start_time
print(f"Temps d'entraînement total : {total_time:.2f} secondes")

# Configuration pour la visualisation de l'accuracy et du total loss ensemble pour le modèle CNN TL
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(history.history['loss'], label='Training Loss', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(history.history['accuracy'], label='Training Accuracy', color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.title('CNN Training Loss and Accuracy')
plt.show()

# Validation Loss et Validation Accuracy sur le même graphique pour le modèle CNN TL
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation Loss', color=color)
ax1.plot(history.history['val_loss'], label='Validation Loss', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Validation Accuracy', color=color)
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.title('CNN Validation Loss and Validation Accuracy')
plt.show()

# Training Loss et Validation Loss sur le même graphique pour le modèle CNN TL
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss', color='tab:orange')
plt.plot(history.history['val_loss'], label='Validation Loss', color='tab:green')
plt.title('CNN Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Prédiction des classes pour les données de test avec le modèle CNN TL
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Génération de la matrice de confusion
cf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
df_cm = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)

# Affichage de la matrice de confusion
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('CNN Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Affichage du rapport de classification
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))