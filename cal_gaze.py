
import os
import sys
import numpy as np
import math
import pandas as pd

def merge_head_predict():
	root_dir='predict/'
	file_names=['-1_aug_pred_head.txt','-0.5_aug_pred_head.txt','-0.1_aug_pred_head.txt','0_aug_pred_head.txt',
			'0.1_aug_pred_head.txt','0.5_aug_pred_head.txt','1_aug_pred_head.txt']
	big_table={}
	name_list = []
	for filename in file_names:
		with open(root_dir+filename,'r') as f:
				lines = f.read()
		lines = lines.split('\n')

		for i in range(int(len(lines) // 3)):
			name, lo, la = lines[i * 3], float(lines[i * 3 + 1]), float(lines[i * 3 + 2])
			if name in big_table:
				big_table[name].append([lo,la])
			else:
				big_table[name]=[[lo,la]]
			if name not in name_list:
				name_list.append(name)
		print(filename, len(name_list))
	print('all data:',len(name_list))
	
	predict_lola = []
	for name in name_list:
		lolas=big_table[name]
		los = [x[0] for x in lolas]
		las = [x[1] for x in lolas]
		best_lo = np.median(los)
		best_la = np.median(las)
		predict_lola.append([best_lo, best_la])

	with open('predict_head.txt',"w") as f:
		for i in range(len(name_list)):
			f.write(name_list[i] + "\n")
			f.write("%0.3f\n" % (predict_lola[i][0]))
			f.write("%0.3f\n" % (predict_lola[i][1]))

def merge_eye_predict():

	root_dir='predict/'
	file_names = ['pred_eye.txt', 'aug_pred_eye.txt']
	big_table = {}
	name_list = []
	for filename in file_names:
		with open(root_dir + filename, 'r') as f:
			lines = f.read()
		lines = lines.split('\n')
		
		for i in range(int(len(lines) // 3)):
			name, lo, la = lines[i * 3], float(lines[i * 3 + 1]), float(lines[i * 3 + 2])
			if name in big_table:
				big_table[name].append([lo, la])
			else:
				big_table[name] = [[lo, la]]
			if name not in name_list:
				name_list.append(name)
		print(filename, len(name_list))
	print('all data:', len(name_list))
	
	predict_lola = []
	for name in name_list:
		lolas = big_table[name]
		los = [x[0] for x in lolas]
		las = [x[1] for x in lolas]
		best_lo = np.mean(los)
		best_la = np.mean(las)
		predict_lola.append([best_lo, best_la])
	
	with open('predict_eye.txt', "w") as f:
		for i in range(len(name_list)):
			f.write(name_list[i] + "\n")
			f.write("%0.3f\n" % (predict_lola[i][0]))
			f.write("%0.3f\n" % (predict_lola[i][1]))

def cal_gaze_equation(head_lo, head_la, eye_lo, eye_la):
	cA = math.cos(head_lo / 180 * np.pi)
	sA = math.sin(head_lo / 180 * np.pi)
	cB = math.cos(head_la / 180 * np.pi)
	sB = math.sin(head_la / 180 * np.pi)
	cC = math.cos(eye_lo / 180 * np.pi)
	sC = math.sin(eye_lo / 180 * np.pi)
	cD = math.cos(eye_la / 180 * np.pi)
	sD = math.sin(eye_la / 180 * np.pi)
	g_x = - cA * sC * cD + sA * sB * sD - sA * cB * cC * cD
	g_y = cB * sD + sB * cC * cD
	g_z = sA * sC * cD + cA * sB * sD - cA * cB * cC * cD
	gaze_lo = math.atan2(-g_x, -g_z) * 180.0 / np.pi
	gaze_la = -math.asin(g_y) * 180.0 / np.pi
	
	return gaze_lo, gaze_la
	
def cal_gaze_from_HeadEye():

	with open('predict/predict_head.txt', 'r') as f:
		lines = f.read()
	lines=lines.split('\n')
	head = []
	for i in range(int(len(lines) // 3)):
		name, lo, la = lines[3*i], lines[3*i + 1], lines[3*i + 2]
		head.append({'name': name, 'lo': lo, 'la': la})
	print('head:',len(head))
	
	with open('predict/predict_eye.txt', 'r') as f:
		lines = f.read()
	lines=lines.split('\n')
	eye = []
	for i in range(int(len(lines) // 3)):
		name, lo, la = lines[3*i], lines[3*i + 1], lines[3*i + 2]
		eye.append({'name': name, 'lo': lo, 'la': la})
	print('eye:',len(eye))

	gaze=[]
	for h, e in zip(head, eye):
		name=h['name']
		if name!=e['name']:
			print('fuck you!!!')
			break
		head_lo = float(h['lo'])
		head_la = float(h['la'])
		eye_lo = float(e['lo'])
		eye_la = float(e['la'])

		gaze_lo, gaze_la = cal_gaze_equation(head_lo, head_la, eye_lo, eye_la)

		gaze.append({'name': name, 'lo': gaze_lo, 'la': gaze_la})

	print('gaze len:',len(gaze))
	with open('predict/predict_gaze.txt', "w") as f:
		for line in gaze:
			f.write(str(line['name']))
			f.write("%0.3f\n" % line['lo'])
			f.write("%0.3f\n" % line['la'])
			
if __name__ == '__main__':
	merge_head_predict()
	merge_eye_predict()
	cal_gaze_from_HeadEye()