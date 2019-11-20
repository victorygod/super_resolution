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
        # output = tf.nn.relu6(x)/3.0-1.0
        output = tf.keras.layers.Lambda(tf.nn.relu)(x)

        model = tf.keras.Model(inputs = inputs, outputs = output)
        model.compile(optimizer="adam", loss=tf.keras.losses.MAE, metrics=[self.PSNR])

        return model

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
        import cv2
        test_data = DataFromUrl("mtid_size_url_data_1", (128, 128))
        ori_images = test_data.get_data(10, resize=False).astype(np.float32)
        img_size = (ori_images.shape[2], ori_images.shape[1])

        noise_imgs = []
        for ori_img in ori_images:
            ratio = 2#random.choice([1.5, 2, 3, 4])
            #todo noise
            img = cv2.resize(ori_img, (int(img_size[0]*ratio), int(img_size[1]*ratio)), interpolation=cv2.INTER_CUBIC)
            noise_imgs.append(img)
        noise_imgs = np.array(noise_imgs)/255
        pred_images = model.predict(noise_imgs)


        self.save_images(pred_images, "_pred")
        self.save_images(noise_imgs, "_input")


if __name__=="__main__":
    net = REDCNN()
    # net.train()
    net.predict()
