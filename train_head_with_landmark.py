#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import numpy as np
import math
import cv2
import time

from sklearn import linear_model
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pickle
from itertools import chain
from sklearn.metrics import mean_absolute_error

def load_gt(filename):
	ret = {}
	with open(filename,"r") as f:
		while True:
			line = f.readline()
			if not line:
				break
			line = line.strip("\n")+".png"
			lo = float(f.readline().strip("\n"))
			la = float(f.readline().strip("\n"))
			ret[line] = np.array([lo,la],dtype=np.float32)
	return ret

def train_lgb(x_train, x_valid, y_train, y_valid , name='0'):
	t1=time.time()
	print('begin to train model_',name)
	d_train = lgb.Dataset(x_train, y_train)
	d_valid = lgb.Dataset(x_valid, y_valid)

	watch_list = [d_train, d_valid]

	params = {
		'learning_rate':0.02,
		'objective':'regression_l2', #'regression_l1'
		'metric':{'l1','l2'},
		'num_leaves':64,
		'subsample':0.7,
		'lambda_l1':0.01,
		'colsample_bytree':0.6,
		'nthread':8,
		#'device_type':'gpu',
	}

	model = lgb.train(params, d_train, 10000, watch_list,early_stopping_rounds=100,verbose_eval=10)
	#x=np.vstack([x_train,x_valid])
	#y=y_train.extend(y_valid)
	#print('x-y',x.shape,len(y))
	#model = lgb.train(params,lgb.Dataset(x,y),num_boost_round=10000)
	print('finished train model_',name,' time cost:',time.time()-t1)
	joblib.dump(model,'ml_model/'+name+'lgb.model',compress=3)
	return model

def get_vertical(points):
	A=np.array(points[36])
	B=np.array(points[33])
	C=np.array(points[45])
	a=B-A
	b=B-C
	c=np.array([a[1]*b[2]-b[1]*a[2],a[2]*b[0]-b[2]*a[0],a[0]*b[1]-b[0]*a[1]])
	c=c/(np.sqrt(c.dot(c)))

	return list(c)

def train(data_dir):
	t1=time.time()
	head_label = load_gt(os.path.join(data_dir, "head_label.txt"))
	X=[];Y=[]; name_list=[]
	data = joblib.load(os.path.join(data_dir, 'train_landmark.pkl'))
	for img_name, feat_ in data.items():   #feat[0] 2d(68,2) #feat[1] 3d(68,3)
		feat = list(chain(*(feat_[0])))
		feat.extend(list(chain(*(feat_[1]))))
		feat = np.array(feat)
		v=get_vertical(feat_[1])
		feat=list(feat)
		feat.extend(v)
		feat = np.array(feat)
		X.append(feat)
		Y.append(head_label[img_name])
	print('finished to prepare data, time cost:',time.time()-t1)
	X=np.array(X,dtype=np.float32)
	print(X.shape)
	x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.05,random_state=1)
	x_train=np.array(x_train,dtype=np.float32)
	x_valid = np.array(x_valid, dtype=np.float32)
	
	lo_train=[y[0] for y in y_train]
	lo_valid=[y[0] for y in y_valid]
	lo_model=train_lgb(x_train, x_valid,lo_train, lo_valid,'lo')
	predict_lo = lo_model.predict(x_valid)
	print('shape:',predict_lo.shape,x_valid.shape)
	
	la_train=[y[1] for y in y_train]
	la_valid=[y[1] for y in y_valid]
	la_model=train_lgb(x_train, x_valid,la_train, la_valid,'la')
	predict_la = la_model.predict(x_valid)
	
	err = mean_absolute_error(lo_valid,predict_lo) + mean_absolute_error(la_valid, predict_la)
	print('err',err)

def predict(data_dir):
	test= joblib.load(os.path.join(data_dir, 'test_landmark.pkl'))
	print('fan:',len(test))
	X=[];name_list=[]
	for img_name, feat_ in test.items():   #feat[0] 2d(68,2) #feat[1] 3d(68,3)
		feat = list(chain(*(feat_[0])))
		feat.extend(list(chain(*(feat_[1]))))
		v=get_vertical(feat_[1])
		feat=list(feat)
		feat.extend(v)
		feat = np.array(feat)
		X.append(feat)
		name_list.append(img_name)
		
	lo_model=joblib.load('ml_model/lolgb.model')
	la_model=joblib.load('ml_model/lalgb.model')
	
	predict_lo=lo_model.predict(X)
	predict_la=la_model.predict(X)
	
	with open('predict/lgb_pred_head.txt',"w") as f:
		for i in range(len(name_list)):
			f.write(name_list[i].split(".")[0]+"\n")
			f.write("%0.3f\n" % predict_lo[i])
			f.write("%0.3f\n" % predict_la[i])
	print('fan lgb predict done...')


if __name__ == '__main__':
	train('sample_data')
	predict('sample_data')

