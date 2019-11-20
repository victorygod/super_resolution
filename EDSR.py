import tensorflow as tf
from network import SISR
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def depth_to_space(ps_conv):
    x = tf.depth_to_space(ps_conv, 2)
    return x

class EDSR(SISR):
    def __init__(self):
        super(EDSR, self).__init__(model_name="EDSR")

    def build_model(self, istraining=True):
        inputs = tf.keras.Input(shape=(None, None, 3))
        
        features = 256
        x = tf.keras.layers.Conv2D(features, (3, 3), activation='relu', padding="same")(inputs)
        conv1 = x


        for i in range(32):
            #x = self.resBlock(x, features=features, scale = 0.1)
            x = tf.keras.layers.Lambda(self.resBlock)(x)
        # pixel shuffle
        x = tf.keras.layers.Conv2D(features, (3, 3), padding = "same")(x)
        x = tf.keras.layers.Add()([conv1, x])
        
        ps_conv = tf.keras.layers.Conv2D(features*4, (3, 3), padding="same", activation = 'relu')(x)
        x = tf.keras.layers.Lambda(depth_to_space)(ps_conv)
        output = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation='relu')(x)


        #output = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding="same", activation='relu')(x)
        
        #output = tf.nn.relu6(x)/3.0-1.0

        model = tf.keras.models.Model(inputs = inputs, outputs = output)
        model.compile(optimizer="adam", loss=tf.keras.losses.mean_absolute_error, metrics=[self.PSNR])


        return model


    def resBlock(self, inputs, features=256, scale = 0.1):
        c1 = tf.keras.layers.Conv2D(features, (3, 3), padding = "same", activation = 'relu')(inputs)
        c2 = tf.keras.layers.Conv2D(features, (3, 3), padding = "same")(c1)
        c2 = tf.keras.layers.Add()([inputs, c2 * scale])
        return c2

    def predict(self):
        assert os.path.exists(self.restore_path), self.restore_path+" not exits!"

        model = self.build_model(False)
        model.load_weights(self.restore_path)
        print("restored from", self.restore_path)
        
        # files = os.listdir("inputs")
        # for file in files:
        #     im = 
        from dataloader import DataFromUrl
        import numpy as np
        test_data = DataFromUrl("mtid_size_url_data_1", (128, 128))
        ori_images = test_data.get_data(10, resize=False).astype(np.float32)/127.5 - 1

        pred_images = model.predict(ori_images)

        self.save_images(pred_images, "_pred")

if __name__=="__main__":
    net = EDSR()
    # net.train(preprocessor=net.ratio_preprocess)
    net.predict()
