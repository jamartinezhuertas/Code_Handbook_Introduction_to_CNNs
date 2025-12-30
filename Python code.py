# Cargamos los recursos de Python necesarios
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models,
      utils
# Definimos que CIFAR100 será nuestra base de datos
(x_train, y_train), (x_test, y_test) = 
      datasets.cifar100.load_data()
# Comprobamos su dimensionalidad
print("Dimensionalidad entrenamiento:", x_train.shape)
print("Dimensionalidad test:", x_test.shape)
# Cambiamos la métrica de los píxeles de las imágenes
x_train = x_train / 255.0
x_test = x_test / 255.0
# Especificamos que las etiquetas son categorías
y_train = utils.to_categorical(y_train, 100)
y_test = utils.to_categorical(y_test, 100)
# Obtenemos un gráfico con imágenes aleatorias de CIFAR100
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10,
      10))
indices = np.random.choice(range(50000), 25, replace=False)
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[indices[i]])
    ax.axis('off')
plt.show()
# Especificamos nuestro modelo
model = models.Sequential([
      layers.Conv2D(32, (3, 3), padding='same',
      activation='relu', input_shape=(32, 32, 3)),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), padding='same',
      activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3, 3), padding='same',
      activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(256, (3, 3), padding='same',
      activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Flatten(),
      layers.Dropout(0.5),
      layers.Dense(512, activation='relu'),
      layers.Dense(100, activation='softmax')
      ])
# Comprobamos las características del modelo especificado
model.summary()
# Establecemos sus características y estimamos el modelo
model.compile(optimizer='rmsprop',
      loss='categorical_crossentropy',
      metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=30,
      batch_size=128, validation_split=0.2)
# Predecimos en el conjunto de test 
predictions = np.argmax(model.predict(x_test), axis=1)
# Evaluamos el rendimiento en el conjunto de test
accuracy = np.mean(predictions == np.argmax(y_test,
      axis=1))
print("Rendimiento:", accuracy)
