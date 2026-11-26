import os
import numpy as np
import tensorflow as tf
from PIL import Image

NOMBRES_CLASES = [
    "Perro (cane)", "Caballo (cavallo)", "Elefante (elefante)", 
    "Mariposa (farfalla)", "Gallina (gallina)", "Gato (gatto)", 
    "Vaca (mucca)", "Oveja (pecora)", "Araña (ragno)", "Ardilla (scoiattolo)"
]

@tf.keras.utils.register_keras_serializable()
def dummy_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

class CNNInferencia:
    def __init__(self, ruta_modelo="modelos/modelo_cnn_animales.keras"):
        self.ruta_modelo = ruta_modelo
        self.modelo = None
        self.cargar_modelo()

    def cargar_modelo(self):
        if os.path.exists(self.ruta_modelo):
            try:
                self.modelo = tf.keras.models.load_model(self.ruta_modelo, compile=False)
            except Exception as e:
                print(f"Error al cargar el modelo: {e}")
                self.modelo = None
        else:
            print(f"Archivo de modelo no encontrado en {self.ruta_modelo}")

    def preprocesar_imagen(self, imagen_pil, target_size=(128, 128)):
        img = imagen_pil.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch

    def predecir(self, imagen_pil):
        if self.modelo is None:
            return None, None
        
        input_tensor = self.preprocesar_imagen(imagen_pil)
        predicciones = self.modelo.predict(input_tensor, verbose=0)[0]
        
        idx_max = np.argmax(predicciones)
        clase_predicha = NOMBRES_CLASES[idx_max]
        confianza = float(predicciones[idx_max] * 100)
        
        resultados = {
            "clase": clase_predicha,
            "confianza": confianza,
            "probabilidades": {NOMBRES_CLASES[i]: float(predicciones[i] * 100) for i in range(len(NOMBRES_CLASES))}
        }
        return resultados