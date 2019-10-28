import numpy as np
import os
import random
from random import shuffle
import cv2
import urllib3
import urllib3.contrib.pyopenssl
import certifi
import socket
socket.setdefaulttimeout(30)

class DataFromUrl():
	def __init__(self, dict_path, img_shape=(128, 128)):
		self.file_counter = 0
		self.files = self._load_dict(dict_path)
		self.file_num = len(self.files)
		self.cache_path = "image_cache/"
		self.img_shape = img_shape
		if not os.path.isdir(self.cache_path):
			os.mkdir(self.cache_path)
		shuffle(self.files)


	def _load_dict(self, dict_path):
		files = []
		with open(dict_path, "r") as f:
			for line in f:
				segs = line.strip().split("\t")
				if len(segs)!=2:
					continue
				mtid_size, url = segs
				files.append({"name": mtid_size+".jpg", "url":url})
		return files

	def get_data(self, batch_size, resize=False):
		if self.file_counter > self.file_num - 1:
			print("Data exhausted, Re Initialize")
			self.file_counter = 0
			shuffle(self.files)
			return None

		batch_data = []
		i=0
		while i<batch_size:
			i+=1
			try:
				if self.file_counter > self.file_num - 1:
					break
				
				file = self.files[self.file_counter]
				image_path = os.path.join(self.cache_path, file["name"])
				self.file_counter+=1
				for _ in range(5):
					try:
						if not os.path.exists(image_path):
							header = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36"}
							http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
							r = http.request("GET", file["url"], None, header)
							with open(image_path, "wb") as f:
							    f.write(r.data)
							r.release_conn()
						break
					except socket.timeout:
						print("retry download: ", file["name"], file["url"])
				image = cv2.imread(image_path)
				if image.shape[0]<self.img_shape[0] or image.shape[1]<self.img_shape[1]:
					i-=1
					print(file["name"], " too small, size:", image.shape)
					continue

				if resize:
					image = cv2.resize(image, self.img_shape)
				else:
					offset = (image.shape[0]-self.img_shape[0], image.shape[1]-self.img_shape[1])
					x, y = random.randint(0, offset[0]), random.randint(0, offset[1])
					image = image[x:x+self.img_shape[0], y:y+self.img_shape[1], :]
				# for _ in range(augment_times):
				# 	new_image = aug_image(image)
				batch_data.append(image)
		
			except Exception as e:
				print("error: ", e)
				print("file name: ", image_path)
				i-=1

		if len(batch_data)==0:
			return None

		return np.array(batch_data)
