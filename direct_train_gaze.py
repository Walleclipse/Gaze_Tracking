
import warnings
import os,sys
import numpy as np
import random
import cv2
from sklearn.model_selection import train_test_split
from time import time
import math
from imgaug import augmenters as iaa
import keras
import tensorflow as tf
from keras import applications as KA
from keras import backend as K
from keras.models import Model,Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, add, Lambda, concatenate,ZeroPadding2D, AveragePooling2D
from keras import optimizers
import keras.backend.tensorflow_backend as KTF
from shufflenetv2 import ShuffleNetV2
from sklearn.metrics import mean_absolute_error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
session = tf.Session(config=config)
KTF.set_session(session)

data_dir = 'sample_data'
CHECKPOINT_FOLDER = 'keras_models'
BATCH_SIZE =64
EPOCHS = 50
alpha=0.4

def load_gt(filename):
	ret = {}
	with open(filename, "r") as f:
		while True:
			line = f.readline()
			if not line:
				break
			line = line.strip("\n") + ".png"
			lo = float(f.readline().strip("\n"))
			la = float(f.readline().strip("\n"))
			ret[line] = np.array([lo, la], dtype=np.float32)
	return ret

imgaugment = iaa.SomeOf((0, 5), [
	iaa.Noop(),
	iaa.Sometimes(0.2,
		iaa.CropAndPad(percent=(-0.05, 0.05)),  # random crops
	),
	iaa.GaussianBlur(sigma=(0, 1.8)),
	iaa.Sometimes(0.2,
				  iaa.AverageBlur(k=(1, 3))
	),
	iaa.Sometimes(0.05,
		iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.9, 1.1))
	),
	iaa.ContrastNormalization((0.8, 1.22)),
	iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),
], random_order=True)

imgflip = iaa.Fliplr(1)


def gazeImageLoader(img_list, root_dir='data/train', batch_size=BATCH_SIZE, img_size=224, train_mode=False,
					imgaug=False):
	gaze_label = None
	if train_mode:
		gaze_label = load_gt(os.path.join(root_dir, "gaze_label.txt"))
	L = len(img_list)
	while True:
		if train_mode:
			np.random.shuffle(img_list)
		batch_start = 0
		batch_end = batch_size
		
		while batch_end <= L:
			x_train, y_train = [], []
			for img_name in img_list[batch_start:batch_end]:
				imgh = cv2.imread(os.path.join(root_dir, 'head', img_name),
								  cv2.IMREAD_GRAYSCALE)  # cv2.IMREAD_GRAYSCALE
				mid_x, mid_y = imgh.shape[0] // 2, imgh.shape[1] // 2
				imgh = cv2.resize(imgh, (2 * mid_x, 2 * mid_y))
				imgh = imgh[mid_x - 200:mid_x + 200, mid_y - 200:mid_y + 200]
				imgh = cv2.resize(imgh, (img_size, img_size))
				
				imgl = cv2.imread(os.path.join(root_dir, 'l_eye', img_name),
								  cv2.IMREAD_GRAYSCALE)  # , cv2.IMREAD_GRAYSCALE
				imgl = cv2.resize(imgl, (120, 80))
				mid_x, mid_y = imgl.shape[0] // 2, imgl.shape[1] // 2
				imgl = imgl[mid_x - 35:mid_x + 35, mid_y - 55: mid_y + 55]
				imgl = cv2.resize(imgl, (img_size, img_size))
				
				imgr = cv2.imread(os.path.join(root_dir, 'r_eye', img_name),
								  cv2.IMREAD_GRAYSCALE)  # , cv2.IMREAD_GRAYSCALE
				imgr = cv2.resize(imgr, (120, 80))
				mid_x, mid_y = imgr.shape[0] // 2, imgr.shape[1] // 2
				imgr = imgr[mid_x - 35:mid_x + 35, mid_y - 55: mid_y + 55]
				imgr = cv2.resize(imgr, (img_size, img_size))
				img = np.stack([imgl, imgr, imgh], axis=2)
				
				if train_mode and imgaug and random.random() < 0.5:
					img = imgaugment.augment_image(img)
					if random.random() < 0.5:
						img = imgflip.augment_image(img)
						img[:, :, 0], img[:, :, 1] = img[:, :, 1], img[:, :, 0]
						gaze_label[img_name] = [- gaze_label[img_name][0], gaze_label[img_name][1]]
				
				if train_mode and imgaug and random.random() < 0.3:
					ss = random.randint(1, L)
					imghs = cv2.imread(os.path.join(root_dir, 'head', img_list[ss]),
									   cv2.IMREAD_GRAYSCALE)  # cvEAD_GRAYSCALE
					mid_x, mid_y = imghs.shape[0] // 2, imghs.shape[1] // 2
					imghs = cv2.resize(imghs, (2 * mid_x, 2 * mid_y))
					
					imghs = imghs[mid_x - 200:mid_x + 200, mid_y - 200:mid_y + 200]
					imghs = cv2.resize(imghs, (img_size, img_size))
					
					imgls = cv2.imread(os.path.join(root_dir, 'l_eye', img_list[ss]),
									   cv2.IMREAD_GRAYSCALE)  # , cvIMRRAYSCALE
					imgls = cv2.resize(imgls, (120, 80))
					mid_x, mid_y = imgls.shape[0] // 2, imgls.shape[1] // 2
					imgls = imgls[mid_x - 35:mid_x + 35, mid_y - 55: mid_y + 55]
					imgls = cv2.resize(imgls, (img_size, img_size))
					
					imgrs = cv2.imread(os.path.join(root_dir, 'r_eye', img_list[ss]),
									   cv2.IMREAD_GRAYSCALE)  # D_GRAYSCALE
					imgrs = cv2.resize(imgrs, (120, 80))
					mid_x, mid_y = imgrs.shape[0] // 2, imgrs.shape[1] // 2
					imgrs = imgrs[mid_x - 35:mid_x + 35, mid_y - 55: mid_y + 55]
					imgrs = cv2.resize(imgrs, (img_size, img_size))
					imgs = np.stack([imgls, imgrs, imghs], axis=2)
					ww = np.random.beta(alpha, alpha)
					img = ww * img + (1 - ww) * imgs
					gaze_label[img_name] = ww * np.array(gaze_label[img_name]) + (1 - ww) * np.array(
						gaze_label[img_list[ss]])
				
				if imgaug and not train_mode:
					img = imgflip.augment_image(img)
					img[:, :, 0], img[:, :, 1] = img[:, :, 1], img[:, :, 0]
				
				img = img / 255
				x_train.append(img)
				if train_mode == True:
					y_train.append(gaze_label[img_name])
			
			if train_mode:
				randnum = random.randint(0, 100)
				random.seed(randnum)
				random.shuffle(x_train)
				random.seed(randnum)
				random.shuffle(y_train)
				yield np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32)
			else:
				yield np.array(x_train, dtype=np.float32)
			batch_start += batch_size
			batch_end += batch_size

def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0):
	Gt = tf.reshape(y_true, [-1, 2])
	Pt = tf.reshape(y_pred, [-1, 2])
	x = Pt - Gt
	x = tf.abs(x)
	c = w * (1.0 - math.log(1.0 + w / epsilon))
	losses = tf.where(
		tf.greater(w, x),
		w * tf.log(1.0 + x / epsilon),
		x - c
	)
	
	loss = tf.reduce_mean(tf.reduce_sum(losses, axis=-1), axis=0)
	return loss

def train_gaze():
	print('direct train_gaze.................')
	root_dir = 'sample_data'
	val_dir = 'sample_data/val_data'
	img_list = os.listdir(os.path.join(root_dir, 'head'))
	# val_list = os.listdir(val_dir + '/head')
	print('loaded data:', len(img_list))
	# train_list = list(set(img_list) - set(val_list))
	train_list, val_list = train_test_split(img_list, test_size=0.05, random_state=1)
	print('train data set:', len(train_list), ' val data set:', len(val_list))
	trainloader = gazeImageLoader(train_list, root_dir=root_dir, batch_size=BATCH_SIZE, img_size=224, train_mode=True,
								  imgaug=True)
	valloader = gazeImageLoader(val_list, root_dir=root_dir, batch_size=BATCH_SIZE, img_size=224, train_mode=True,
								imgaug=False)
	STEPS_PER_EPOCH = int(len(train_list) // BATCH_SIZE)
	VAL_STEPS = int(len(val_list) / BATCH_SIZE)
	base_model = KA.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), classes=1000)
	x = base_model.output
	x = Flatten(name='flatten1')(x)
	x = Dense(1000, activation='relu', name='fl1')(x)
	out = Dense(2, activation='linear', name='out_layer')(x)
	model = Model(base_model.input, out)
	# model = load_model(CHECKPOINT_FOLDER+'/gaze_model.h5')
	print(model.summary())
	
	ckpt = keras.callbacks.ModelCheckpoint(CHECKPOINT_FOLDER + '/gaze_model.h5', monitor='val_loss', verbose=1,
										   save_best_only=True, mode='auto', period=1)
	stp = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
	tb = keras.callbacks.TensorBoard(log_dir=CHECKPOINT_FOLDER + '/gazelogs', histogram_freq=0, write_graph=True,
									 write_images=False, embeddings_freq=0, embeddings_layer_names=None,
									 embeddings_metadata=None)
	model.compile(optimizer='adam', loss=wing_loss, metrics=['mse', 'mae'])  # optimizers.SGD(lr=0.0001,momentum=0.9)
	
	model.fit_generator(trainloader, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
						validation_data=valloader, validation_steps=VAL_STEPS,
						verbose=1, callbacks=[ckpt, stp, tb], workers=4, pickle_safe=True, initial_epoch=0)

def predict_gaze():
	batch_size = 10  # must be the multiple of len(img_list)
	root_dir = 'sample_data'
	img_list = os.listdir(os.path.join(root_dir, 'head'))
	
	TEST_STEPS = int(len(img_list) / batch_size)
	gazemodel = load_model(CHECKPOINT_FOLDER + '/gaze_model.h5')
	
	gazeloader = gazeImageLoader(img_list, root_dir=root_dir, batch_size=batch_size, img_size=224, imgaug=False)
	predgaze = gazemodel.predict_generator(gazeloader, steps=TEST_STEPS, workers=4)
	
	with open('predict/' + 'direct_pred_gaze.txt', "w") as f:
		for i in range(len(img_list)):
			f.write(img_list[i].split(".")[0] + "\n")
			f.write("%0.3f\n" % predgaze[i][0])
			f.write("%0.3f\n" % predgaze[i][1])
	print(' gaze predict done...')

if __name__ == '__main__':
	train_gaze()
	predict_gaze()