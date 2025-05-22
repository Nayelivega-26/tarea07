import tensorflow as tf
import numpy as np

# Datos ficticios: [color_rojo, color_amarillo, tamaño_grande, peso_en_kg]
# Ejemplo: [1, 0, 1, 0.2] → rojo, no amarillo, grande, 0.2 kg
x_train = np.array([
    [1, 0, 1, 0.2],  # manzana
    [0, 1, 0, 0.15], # plátano
    [1, 0, 0, 0.05], # cereza
    [0, 1, 1, 0.3],  # mango
    [1, 0, 1, 0.25], # manzana
    [0, 1, 0, 0.1],  # plátano
])

# Etiquetas: 0 = manzana, 1 = plátano, 2 = cereza, 3 = mango
y_train = np.array([0, 1, 2, 3, 0, 1])

# Crear el modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 frutas
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar
model.fit(x_train, y_train, epochs=100, verbose=0)

# Probar con una nueva fruta: roja, grande, 0.2 kg → debería ser manzana
test = np.array([[1, 0, 1, 0.2]])
prediction = model.predict(test)

# Mostrar predicción
frutas = ['Manzana', 'Plátano', 'Cereza', 'Mango']
print("Fruta predicha:", frutas[np.argmax(prediction)])
