import tensorflow as tf
from network import SISR
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class EDSR(SISR):
    def __init__(self):
        super(EDSR, self).__init__(model_name="EDSR")

    def build_model(self, istraining=True):
        inputs = tf.keras.Input(shape=(None, None, 3))
        
        features = 256
        x = tf.keras.layers.Conv2D(features, (3, 3), activation='relu', padding="same")(inputs)
        
        for i in range(32):
            x = self.resBlock(x, features=features, scale = 0.1)

        # pixel shuffle
        r = 2
        ps_conv = tf.keras.layers.Conv2D(features*r**2, (3, 3), padding="same", activation = 'relu')(x)
        x = tf.depth_to_space(ps_conv, r)
        x = tf.keras.layers.Conv2D(3, (3, 3), padding="same")(x)

        # x = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding="same")(x)
        
        output = tf.nn.relu6(x)/3.0-1.0

        model = tf.keras.Model(inputs = inputs, outputs = output)
        model.compile(optimizer="adam", loss=tf.keras.losses.MAE, metrics=[self.PSNR])

        return model


    def resBlock(self, inputs, features=256, scale = 0.1):
        c1 = tf.keras.layers.Conv2D(features, (3, 3), padding = "same")(inputs)
        x = tf.nn.relu(c1)
        c2 = tf.keras.layers.Conv2D(features, (3, 3), padding = "same")(x)
        c2 = tf.keras.layers.Add()([inputs, c2 * scale])
        return c2

if __name__=="__main__":
    net = EDSR()
    net.train(preprocessor=net.ratio_preprocess)
