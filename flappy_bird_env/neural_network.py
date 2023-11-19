import tensorflow as tf
from tensorflow.python.keras import layers, models

def build_model(input_shape, num_actions):
    model = models.Sequential()

    # Normalización de las observaciones
    model.add(layers.BatchNormalization(input_shape=input_shape))

    # Capas convolucionales
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Aplanar para conectar con capas completamente conectadas
    model.add(layers.Flatten())

    # Capas completamente conectadas
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_actions, activation='softmax'))  # Ajustar a num_actions

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Mostrar resumen del modelo
    model.summary()

    return model
