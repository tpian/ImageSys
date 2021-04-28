from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import random
import glob
import matplotlib.pyplot as plt

input_dir = "data_prepare/ReductionPic/Animals/val_resize1"
target_dir = "data_prepare/ReductionPic/Animals/val_gaussian_noise_resize1"

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

with tf.name_scope("load_images"):
    input_paths0 = glob.glob(os.path.join(input_dir, "*.jpg"))#返回所有匹配的文件路径列表
    target_paths0 = glob.glob(os.path.join(target_dir,"*.jpg"))

    img_data = tf.gfile.FastGFile(input_paths0[0], 'rb').read()
    img_data = tf.image.decode_jpeg(img_data)
    flipped1 = tf.image.random_flip_up_down(img_data)

    input_paths = sorted(input_paths0)
    target_paths = sorted(target_paths0)

    decode = tf.image.decode_jpeg

    seed = random.randint(0, 2**31 - 1)


    input_queue = tf.train.string_input_producer(input_paths, shuffle= True ,seed = seed)
    target_queue = tf.train.string_input_producer(target_paths, shuffle= True ,seed = seed)
    #输出字符串到一个输入管道队列，shuffle： bool类型，设置是否打乱样本的顺序。

    reader = tf.WholeFileReader()

    input_paths, inputs0 = reader.read(input_queue)
    target_paths, targets0 = reader.read(target_queue)
    #返回值：张量元组(key,value),其中,key是一个字符串标量张量，文件名,value是一个字符串标量张量，文件内容.

    raw_input0 = decode(inputs0)
    raw_target0 = decode(targets0)
    #JEPG编码转换为unit8张量

    raw_input1 = tf.image.convert_image_dtype(raw_input0, dtype=tf.float32)
    raw_target1 = tf.image.convert_image_dtype(raw_target0, dtype=tf.float32)
    #将一个uint类型的tensor转换为float类型时，该方法会自动对数据进行归一化处理，将数据缩放到0-1范围内

    assertion = tf.assert_equal(tf.shape(raw_input1)[2], 3, message="image does not have 3 channels")
    #tf.assert_equal()如果x，y不一致就抛出异常
    with tf.control_dependencies([assertion]):
        raw_input2 = tf.identity(raw_input1)
        raw_target2 = tf.identity(raw_input2)

    raw_input2.set_shape([None, None, 3])
    raw_target2.set_shape([None, None, 3])

    # break apart image pair and move to range [-1, 1]
    inputs = preprocess(raw_input2)
    targets = preprocess(raw_target2)

# synchronize seed for image operations so that we do the same operations to both
    # input and output images
seed = random.randint(0, 2**31 - 1)
def transform(image):
    r = image
    r = tf.image.random_flip_left_right(r, seed=seed)#随机左右翻转图片
    return r

input_images = transform(inputs)
target_images = transform(targets)

image_resize = tf.image.resize_images(input_images, [256, 256])
print(image_resize)

image_input_batch = tf.train.batch([image_resize], batch_size=20, num_threads=1, capacity=20)
print(image_input_batch)

image_input_de = deprocess(image_input_batch)

image_input_de_u8 = tf.image.convert_image_dtype(image_input_de, dtype=tf.uint8)[0]

iamge_encode = tf.image.encode_jpeg(image_input_de_u8)

image_en_de = tf.image.decode_jpeg(iamge_encode)


with tf.Session() as sess:

    print(flipped1.eval())
    print("___________________________________")
    # plt.show()

    coord = tf.train.Coordinator()

    # 开启读文件的子线程
    threads = tf.train.start_queue_runners(sess, coord=coord)

    # 打印读取的内容
    print(sess.run(image_en_de))


    # 结束子线程
    coord.request_stop()
    # 等待子线程结束
    coord.join(threads)

    print(tf.shape(image_en_de))
    print(tf.shape(flipped1.eval()))

    plt.subplot(1,4,2)
    plt.imshow(flipped1.eval())

    plt.subplot(1,4,1)
    plt.imshow(image_en_de)
    # plt.subplot(1, 4,2 )
    # plt.imshow(image_input_de_u8[1])
    # plt.subplot(1, 4, 3)
    # plt.imshow(image_input_de_u8[2])
    # plt.subplot(1, 4, 4)
    # plt.imshow(image_input_de_u8[3])
    plt.show()




