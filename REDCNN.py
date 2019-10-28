import tensorflow as tf
from network import SISR
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class REDCNN(SISR):
    def __init__(self):
        super(REDCNN, self).__init__(model_name="REDCNN")

    def build_model(self, istraining=True):
        inputs = tf.keras.Input(shape=(None, None, 3))
        
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same")(inputs)
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same")(c1)

        x = tf.keras.layers.MaxPool2D((2, 2))(c1)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same")(x)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same")(c2)

        x = tf.keras.layers.MaxPool2D((2, 2))(c2)
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding="same")(x)

        #todo

        d3 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation = 'relu')(c3)
        d3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same")(d3)
        x = tf.keras.layers.Add()([d3, c2])
        d2 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation = 'relu')(x)
        d2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same")(d2)
        x = tf.keras.layers.Add()([d2, c1])
        d1 = tf.keras.layers.Conv2DTranspose(3, (3, 3), padding="same")(x)
        
        x = tf.keras.layers.Add()([d1, inputs])
        output = tf.nn.relu6(x)/3.0-1.0

        model = tf.keras.Model(inputs = inputs, outputs = output)
        model.compile(optimizer="adam", loss=tf.keras.losses.MAE, metrics=[self.PSNR])

        return model



if __name__=="__main__":
    net = REDCNN()
    net.train()
