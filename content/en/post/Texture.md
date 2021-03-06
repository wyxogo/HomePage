---
title: "纹理表示(Texture)"
date: 2020-07-26T15:30:26+08:00

tags: 
    - Texture
    - K均值
categories: 
    - 计算机视觉
featuredImage: "https://img-blog.csdnimg.cn/20200726121818902.png"
featuredImagePreview: "https://img-blog.csdnimg.cn/20200726121818902.png"
---


纹理是由一些基元以某种方式组合起来，虽然看起来很“乱”，但任然存在一些规律

![](https://img-blog.csdnimg.cn/20200726111125370.png " ")

规则的纹理与不规则的纹理

![](https://img-blog.csdnimg.cn/20200726111644948.png " ")

## 纹理描述
1. 使用高斯偏导核，对图像进行卷积，x方向的偏导得到的是竖直纹理，y方向的偏导得到的是水平纹理
2. 统计各个方向的纹理数量，在图中表示出来，不同的区域映射的是不同的纹理特性
3. 如下图所示进行**K均值聚类**

	![](https://img-blog.csdnimg.cn/20200726112808342.png " ")

4. 距离显示了窗口a的纹理和窗口b的纹理有多么不同。

![](https://img-blog.csdnimg.cn/20200726114915709.png " ")

**对于区块核大小的选择**
在图像中往往不知道选取多大的高斯偏导核来对图像进行描述

![](https://img-blog.csdnimg.cn/20200726120305443.png " ")

通过寻找纹理描述不变的窗口比例来进行比例选择，由小到大不断改变窗口的大小，直至增大的窗口纹理特性不再改变

## 滤波器组

1. 可以描述不同方向，不同类型（边状，条状，点）的纹理特性

	![](https://img-blog.csdnimg.cn/20200726121248438.png " ")

2. 通过设置斜方差矩阵$\Sigma$，改变高斯核的形状

	![](https://img-blog.csdnimg.cn/20200726121818902.png " ")

3. 利用不同的核卷积图像

	![](https://img-blog.csdnimg.cn/20200726122741610.png " ")

4. 将响应结果与纹理匹配

	将对应卷积核的响应结果求均值，所得的结果组成一个7维向量，每个向量对应一个纹理

	![](https://img-blog.csdnimg.cn/20200726123312631.png " ")

5. 使用更高维的向量描述

	![](https://img-blog.csdnimg.cn/20200726124508865.png " ")
	
	![](https://img-blog.csdnimg.cn/20200726152207229.png " ")

6. 纹理检索与分类
	实际运用过程中，将采取的纹理与数据库中的纹理进行对比

	![](https://img-blog.csdnimg.cn/20200726153104806.png " ")

**学习资源：[北京邮电大学计算机视觉——鲁鹏](https://www.bilibili.com/video/BV1nz4y197Qv)**
