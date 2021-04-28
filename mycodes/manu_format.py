from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from shutil import copy

input_dir = '../reduction_20210413/images/outputs'
label_txt = '../labels.txt'
output_base = 'redution_20210413_val_class'

if not os.path.exists(output_base):
    os.mkdir(output_base)

src_names = os.listdir(input_dir)
i = 0

f = open(label_txt, 'r')
for line in f:
    line = line.strip()
    os.makedirs(os.path.join(output_base,line))

problemlist = []
for file_name in src_names:
    i = i+1
    # cv2.imwrite(os.path.join(output_dir2, file_name), img2_crop)
    print("共 %d 张图片，正在处理第 %d 张图片" % (len(src_names),i))
    name,_ = os.path.splitext(os.path.basename(file_name))
    kind = name[0:name.rfind("_")]
    copy(os.path.join(input_dir,file_name),os.path.join(output_base,kind,file_name))

print(problemlist)