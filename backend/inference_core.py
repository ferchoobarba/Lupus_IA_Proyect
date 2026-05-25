import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class LupusDetector:
    def __init__(self, weights_path):
        self.img_size = (224, 224)
        self.model = self._build_model()
        self.model.load_weights(weights_path)
        
    def _build_model(self):
        # Reconstruimos la arquitectura exacta
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=self.img_size + (3,), 
            include_top=False, 
            weights=None # Ya no descargamos ImageNet, usamos nuestros pesos
        )
        
        inputs = tf.keras.Input(shape=self.img_size + (3,))
        # (El Augmentation no es necesario para predecir)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid', name='detector_lupus')(x)
        
        return tf.keras.Model(inputs, outputs)

    def predict(self, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_tensor = tf.expand_dims(img_array, 0)
        
        prediccion = self.model.predict(img_tensor, verbose=0)[0][0]
        return float(prediccion)
