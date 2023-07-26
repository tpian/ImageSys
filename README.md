# 1.介绍
东华大学本科生毕业设计: 动物图像识别及分类网站
# 2.技术
前端：Vue框架  
后端：python编程、TensorFlow深度学习框架（主要使用网络为DCGNN与Inception ResNet）、flash轻量化python后端框架，MySQL数据库存储用户提供的图片与纠正的识别结果
# 3.运行环境
## 前端  
环境：
node.js 14.16.1  
运行：
```
# 安装环境
npm install
# 本地运行 serve with hot reload at localhost:8080
npm run dev
# 修改运行参数在/ImageFront/config/index.js中配置
```
## 后端  
环境：  
```
MySQL 5.7
python 3.5  
CUDA 10.0  
cuDNN 7.6  
Tensorflow 1.13.1  
Flask 0.12.2
Flask-Cors 3.0.10
PyMySQL 0.9.3
opencv-python 4.4.0.42
```
运行：  
```
python /ImageServer/lib/controller.py
# 修改数据库参数在/ImageServer/lib/controller中配置
```
# 4.目录结构
```
├─figures
├─ImageFront 前端页面
│  ├─build
│  ├─config
│  ├─node_modules
│  ├─src
│  └─static
└─ImageServer 后端服务器
    ├─.idea
    ├─classifynet20210416 分类网络参数
    ├─reduction_net  降噪网络参数
    ├─lib  主要后端服务代码
    └─shareclass 工具类
```
# 5.主要页面
主页菜单：无需登录，点击即用  
![Image text](https://github.com/tpian/Image/blob/main/figures/homePage.png)  
识别页面：上传图片，获取识别结果的top5，并点击跳转到百度百科相关介绍，用户能在此页面纠正错误的识别结果并保存到后端数据库中  
![Image text](https://github.com/tpian/Image/blob/main/figures/Classification.png)  
降噪页面：上传图片，获取降噪结果  
![Image text](https://github.com/tpian/Image/blob/main/figures/NoiseRuduction.png)
# 6.深度学习网络结构
## 降噪网络：DCGNN  
整体
![Image text](https://github.com/tpian/Image/blob/main/figures/GAN.png)  
生成器  
![Image text](https://github.com/tpian/Image/blob/main/figures/encoder.png)  
鉴别器  
![Image text](https://github.com/tpian/Image/blob/main/figures/decoder.png)  
## 识别网络：Inception ResNet  
![Image text](https://github.com/tpian/Image/blob/main/figures/ReductionNet.png)
