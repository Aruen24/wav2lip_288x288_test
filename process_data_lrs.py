# _*_ coding:utf-8 _*_
import os
import sys
import cv2
import shutil

#LSR2 data
#src_train = './main/'
#src_pretrain = './pretrain/'
src_train = '/Data2/wyw/LRS2/main/'
src_pretrain = '/Data2/wyw/LRS2/pretrain/'

'''
train_class_names = os.listdir(src_train)
pretrain_class_names = os.listdir(src_pretrain)

#pretrain
# 将pretrain下面也在train下面的文件夹中的内容改名然后移动到train对应的文件夹下面
# 如果没有则创建新的文件夹并将其所有文件移动到该新建文件夹下面

for pretrain_class_name in pretrain_class_names:
    if pretrain_class_name in train_class_names:
        # 移动
        pretrain_files = os.listdir(src_pretrain + pretrain_class_name)
        for pretrain_file in pretrain_files:
            # 重新命名并移动
            new_name = str(int(pretrain_file[:-4]) + 1000).zfill(5) + pretrain_file[-4:]
            os.rename(src_pretrain + pretrain_class_name + '/' + pretrain_file, src_pretrain + pretrain_class_name + '/' + new_name)
            cv2.waitKey(10)
            shutil.move(src_pretrain + pretrain_class_name + '/' + new_name, src_train + pretrain_class_name)
    else:
        # 全部移走
        if not os.path.exists(src_train + pretrain_class_name):
            os.makedirs(src_train + pretrain_class_name)
        # 移动
        pretrain_files = os.listdir(src_pretrain + pretrain_class_name)
        for pretrain_file in pretrain_files:
            shutil.move(src_pretrain + pretrain_class_name + '/' + pretrain_file, src_train + pretrain_class_name)

cv2.waitKey(1000)
'''

# 生成train.txt、val.txt
train_class_names = os.listdir(src_train)
train_class_names_len = len(train_class_names)

# train.txt
with open('/Data2/wyw/LRS2/filelists/' + 'train.txt', 'w') as w:
    for train_class_name in train_class_names[:int(train_class_names_len * 0.88)]:
        files = os.listdir(src_train + train_class_name)
        names = []
        for file in files:
            if file.endswith('.txt'):
                w.write(train_class_name + '/' + file[:-4] + "\n")

# val.txt
with open('/Data2/wyw/LRS2/filelists/' + 'val.txt', 'w') as w:
    for train_class_name in train_class_names[int(train_class_names_len * 0.88): int(train_class_names_len * 0.9)]:
        files = os.listdir(src_train + train_class_name)
        names = []
        for file in files:
            if file.endswith('.txt'):
                w.write(train_class_name + '/' + file[:-4] + "\n")









