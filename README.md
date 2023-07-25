# 介绍
东华大学本科生毕业设计: 动物图像识别及分类网站
# 技术
前端：Vue框架  
后端：python编程、TensorFlow深度学习框架（主要使用网络为DCGNN与Inception ResNet）、flash轻量化python后端框架，MySQL数据库存储用户提供的图片与纠正的识别结果
# 主要页面
主页菜单：无需登录，点击即用  
![Image text](https://github.com/tpian/Image/blob/main/figures/homePage.png)  
识别页面：上传图片，获取识别结果，并点击跳转到百度百科相关介绍，用户能在此页面纠正错误的识别结果并保存到后端数据库中  
![Image text](https://github.com/tpian/Image/blob/main/figures/Classification.png)  
降噪页面：上传图片，获取降噪结果  
![Image text](https://github.com/tpian/Image/blob/main/figures/NoiseRuduction.png)
# 深度学习网络结构
## 降噪网络：DCGNN  
整体
![Image text](https://github.com/tpian/Image/blob/main/figures/GAN.png)  
生成器  
![Image text](https://github.com/tpian/Image/blob/main/figures/encoder.png)  
鉴别器  
![Image text](https://github.com/tpian/Image/blob/main/figures/decoder.png)  
## 识别网络：Inception ResNet  
![Image text](https://github.com/tpian/Image/blob/main/figures/ReductionNet.png)
