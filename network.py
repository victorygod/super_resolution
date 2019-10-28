import tensorflow as tf
from dataloader import DataFromUrl
from abc import ABCMeta,abstractmethod
import cv2
import random
import numpy as np
import os

class SISR(object):
    def __init__(self, model_name, max_iter = 10000000, model_dir="models"):
        self.max_iter = max_iter
        self.model_name = model_name
        self.restore_path = os.path.join(model_dir, model_name + ".h5")
        self.restor_iter = os.path.join(model_dir, model_name + "_iter.npy")
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        self.output_dir = "outputs_"+model_name
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    @abstractmethod
    def build_model(self, istraining=True):
        pass

    def train(self, batch_size=20, img_shape=(96, 96), preprocessor=None):
        if preprocessor is None:
            preprocessor = self.identity_preprocess
            
        model = self.build_model(True)
        if os.path.exists(self.restore_path):
            model.load_weights(self.restore_path)
            print("restored from", self.restore_path)

        start_iter = 0 if not os.path.exists(self.restor_iter) else np.load(self.restor_iter)
        print("start from", start_iter)

        train_data = DataFromUrl("mtid_size_url_data_0", img_shape)
        val_data = DataFromUrl("mtid_size_url_data_1", img_shape)
        for i in range(start_iter, self.max_iter):
            ori_images = train_data.get_data(batch_size, resize=False)
            if ori_images is None:
                continue
            ori_images, noise_images = preprocessor(ori_images)

            loss = model.train_on_batch(noise_images, ori_images)

            if i%20==0:
                print(i, loss)
                model.save_weights(self.restore_path)
                print("model saved in", self.restore_path)
                np.save(self.restor_iter, i)

                ori_images = val_data.get_data(10, resize=False)
                while ori_images is None:
                    ori_images = val_data.get_data(10, resize=False)
                ori_images, noise_images = preprocessor(ori_images)
                outputs = model.predict(noise_images)
                
                self.save_images(outputs, "_pred")
                self.save_images(noise_images, "_input")
                self.save_images(ori_images, "_gt")

    def batch_predict(self, imgs):
        
        assert os.path.exists(self.restore_path), self.restore_path+" not exits!"
        assert len(imgs.shape)==4 and imgs.shape[1]%4==0 and imgs.shape[2]%4==0, "shape not good"

        model = self.build_model(False)
        model.load_weights(self.restore_path)
        print("restored from", self.restore_path)
        
        imgs = imgs.astype(np.float32)/127.5 - 1
        return model.predict(imgs)

    def identity_preprocess(self, ori_imgs):
        img_size = (ori_imgs.shape[2], ori_imgs.shape[1])

        noise_imgs = []
        for ori_img in ori_imgs:
            ratio = 2#random.choice([1.5, 2, 3, 4])
            #todo noise
            img = cv2.resize(ori_img, (int(img_size[0]//ratio), int(img_size[1]//ratio)), interpolation=cv2.INTER_CUBIC)
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
            noise_imgs.append(img)

        return ori_imgs.astype(np.float32)/127.5-1, np.array(noise_imgs).astype(np.float32)/127.5-1

    def ratio_preprocess(self, ori_imgs, ratio=2):
        img_size = (ori_imgs.shape[2], ori_imgs.shape[1])
        noise_imgs = []

        for ori_img in ori_imgs:
            img = cv2.resize(ori_img, (int(img_size[0]//ratio), int(img_size[1]//ratio)), interpolation=cv2.INTER_CUBIC)
            noise_imgs.append(img)  

        return ori_imgs.astype(np.float32)/127.5-1, np.array(noise_imgs).astype(np.float32)/127.5-1

    def save_images(self, imgs, suffix):
        for i, img in enumerate(imgs):
            cv2.imwrite(os.path.join(self.output_dir, str(i)+suffix+".jpg"), (img+1)*127.5)

    def PSNR(self, y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
        It can be calculated as
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
        When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
        However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
        Thus we remove that component completely and only compute the remaining MSE component.
        """
        y_true = (y_true+1.0)/2.0
        y_pred = (y_pred+1.0)/2.0
        return -10. * tf.log(tf.reduce_mean(tf.square(y_pred - y_true))) / tf.log(10.)

