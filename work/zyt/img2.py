import imgaug as ia

from imgaug import augmenters as iaa

import numpy as np
import re
import cv2
import tensorflow as tf
import os.path
def image_augmentation(file_name):
	base_name = os.path.basename(file_name)
	img= cv2.imread(file_name)
	sometimes = lambda aug: iaa.Sometimes(0.5, aug) #建立lambda表达式，
	seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5), # 对50%的图像进行镜像翻转
        sometimes(iaa.Crop(percent=(0, 0.1))), 
	   #对随机的一部分图像做crop操作
       # crop的幅度为0到10%
        sometimes(iaa.Affine(                          #对一部分图像做仿射变换
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},#图像缩放为80%到120%之间
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, #平移±20%之间
            rotate=(-45, 45),   #旋转±45度之间
            shear=(-16, 16),    #剪切变换±16度，（矩形变平行四边形）
            order=[0, 1],   #使用最邻近差值或者双线性差值
            cval=(0, 255),  #全白全黑填充
            mode=ia.ALL    #定义填充图像外区域的方法
        )),
        # 使用下面的0个到4个之间的方法去增强图像。
        iaa.SomeOf((0, 4),
            [
                #用高斯模糊，均值模糊，中值模糊中的一种增强
                iaa.OneOf([
                  	iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)), # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                #锐化处理
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                #浮雕效果
               	iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),
                # 将整个图像的对比度变为原来的一半或者二倍
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                #把像素移动到周围的地方。
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),
                # 扭曲图像的局部区域
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            random_order=True # 随机的顺序把这些操作用在图像上
            )
        ],
         random_order=True # 随机的顺序把这些操作用在图像上
       )
	images_aug = [seq.augment_image(img) for _ in range(64)]#一张图片通过数据增广多出来64张
	for i in range(64):
		dir_file=re.sub(r'\..*$', '', file_name)+'_'+str(i)#提取图片后缀之前的名称，并在原名称后添加i表示是增广的第i张图片
		cv2.imwrite(dir_file+os.path.splitext(file_name)[-1],images_aug[i])
	
	
def read_image(image_dir):#该部分和retrain.py中获取地址的操作几乎一样，因此不在此注释
	sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
	is_root_dir=True
	for sub_dir in sub_dirs:#读取地址
		if is_root_dir:
			is_root_dir = False
			continue
		extensions = sorted(set(os.path.normcase(ext)
                            for ext in ['JPEG', 'JPG', 'jpeg', 'jpg', 'png']))
		file_list = []
		dir_name = os.path.basename(
        	sub_dir[:-1] if sub_dir.endswith('/') else sub_dir)
		if dir_name == image_dir:
			continue
		for extension in extensions:
			file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
			file_list.extend(tf.gfile.Glob(file_glob))
		for file_name in file_list:
			image_augmentation(file_name)
		print("FINISH print"+sub_dir)

