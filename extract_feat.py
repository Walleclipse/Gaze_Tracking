import os
import multiprocessing as mp
import face_alignment
import pickle
import json
from time import time
from sklearn.externals import joblib
from multiprocessing import Pool
import threading
import cv2

def prep(rootDir, dir_list, name=0):
	t1=time()
	fa2 = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=False, flip_input=False,use_cnn_face_detector=True)
	fa3 = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=False, flip_input=False,use_cnn_face_detector=True)

	print('worker %d begin ...'%name)
	face_landmarks_lists = {}
	count=0
	for img_name in dir_list:
		count+=1
		try:
			img=cv2.imread(os.path.join(rootDir, img_name))
			pred2=fa2.get_landmarks(img)
			pred3=fa3.get_landmarks(img)
			try:
				pred2.extend(pred3)
				face_landmarks_lists[img_name]=pred2
			except AttributeError as e:
				print(e, img_name)
				with open('feat_record/'+str(name)+'miss_train_landmark.txt','a') as f:
					f.write(img_name+'\n')
		except FileNotFoundError as e:
			print(e, img_name)
		if count%500==0:
			print('processed train %d , time costs: %f s'%(count, time()-t1))

	joblib.dump(face_landmarks_lists,'feat_record/'+'train_landmark.pkl',compress=3)

if __name__ == '__main__':
	t1 = time()
	rootDir = "sample_data/head"
	dir_list = os.listdir(rootDir)
	prep(rootDir,dir_list)



