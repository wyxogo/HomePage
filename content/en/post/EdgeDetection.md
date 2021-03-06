---
title: "边缘检测(Edge Detection)"
date: 2020-07-10T16:50:26+08:00

tags: 
    - 高斯滤波
    - Canny edge detector
categories: 
    - 计算机视觉
featuredImage: "https://img-blog.csdnimg.cn/20200710170507176.png"
featuredImagePreview: "https://img-blog.csdnimg.cn/20200710170507176.png"
---


## 边缘提取
在大多数时候图像的边缘可以承载大部分的信息，并且提取边缘可以除去很多干扰信息，提高处理数据的效率
## 目标
**识别图像中的突然变化(不连续)**
- 图像的大部分语义信息和形状信息都可以编码在边缘上
- 理想:艺术家使用线条勾勒画(但艺术家也使用对象层次的知识)

## 边缘的种类
- 表面形状的突变
- 深度方向的不连续
- 表面颜色的突变
- 光线阴影的不连续

## 边缘的特征
边缘是图像强度函数中快速变化的地方，变化的地方就存在梯度，对灰度值求导，导数为0的点即为**边界点**

![](https://img-blog.csdnimg.cn/20200709230650310.png " ")

**卷积的导数**

- **偏导数公式：**

$$\frac {\partial f(x,y)}{\partial x}  = \lim_{\varepsilon \rightarrow 0} \frac{f(x+\varepsilon ,y)-f(x,y)}{\varepsilon}$$

- 在卷积中为描述数据，采取 **近似化处理：**
$$\frac {\partial f(x,y)}{\partial x}  \approx  \frac{f(x+1,y)-f(x,y)}{1}$$

显然在x方向的导数就是与该像素自身与右边相邻像素的**差值**

**卷积描述偏导**

使用卷积核处理

![](https://img-blog.csdnimg.cn/20200709232639610.png " ")
对灰度图的x和y方向分别处理后的效果如下图：
![](https://img-blog.csdnimg.cn/20200709233052117.png " ")

**有限差分滤波器（卷积核）**

- **Roberts 算子**
	Roberts 算子是一种最简单的算子，是一种利用局部差分算子寻找边缘的算子。他采用对角线方向相邻两象素之差近似梯度幅值检测边缘。检测垂直边缘的效果好于斜向边缘，定位精度高，对噪声敏感，无法抑制噪声的影响。
	1963年， Roberts 提出了这种寻找边缘的算子。 Roberts 边缘算子是一个 2x2 的模版，采用的是对角方向相邻的两个像素之差。
	Roberts 算子的模板分为水平方向和垂直方向，如下所示，从其模板可以看出， Roberts 算子能较好的增强正负 45 度的图像边缘。
<div>
$$
dx = \left[
 \begin{matrix}
   -1 & 0\\
   0 & 1 \\
  \end{matrix} 
\right]
$$
</div>

<div>
$$
dy = \left[
 \begin{matrix}
   0 & -1\\
   1 & 0 \\
  \end{matrix} 
\right]
$$
</div>


- **Prewitt算子**
	Prewitt 算子是一种一阶微分算子的边缘检测，利用像素点上下、左右邻点的灰度差，在边缘处达到极值检测边缘，去掉部分伪边缘，对噪声具有平滑作用。Prewitt算子适合用来识别噪声较多、灰度渐变的图像。

<div>
$$
dx = \left[
 \begin{matrix}
   1 & 0 & -1\\
   1 & 0 & -1\\
   1 & 0 & -1\\
  \end{matrix} 
\right]
$$
</div>

<div>
$$
dy = \left[
 \begin{matrix}
   -1 & -1 & -1\\
   0 & 0 & 0\\
   1 & 1 & 1\\
  \end{matrix} 
\right]
$$
</div>

- **Sobel算子**
	Sobel算子是一种用于边缘检测的离散微分算子，它结合了高斯平滑和微分求导。Sobel 算子在 Prewitt 算子的基础上增加了权重的概念，认为相邻点的距离远近对当前像素点的影响是不同的，距离越近的像素点对应当前像素的影响越大，从而实现图像锐化并突出边缘轮廓。
<div>
$$
dx = \left[
 \begin{matrix}
   1 & 0 & -1\\
   2 & 0 & -2\\
   1 & 0 & -1\\
  \end{matrix} 
\right]
$$
</div>
<div>
$$
dy = \left[
 \begin{matrix}
   -1 & -2 & -1\\
   0 & 0 & 0\\
   1 & 2 & 1\\
  \end{matrix} 
\right]
$$
</div>

## 图像梯度

$$\nabla f=[\frac{\partial f}{\partial x},\frac{\partial f}{\partial y}]$$

- **梯度指向强度增长最快的方向**

![](https://img-blog.csdnimg.cn/20200710112756581.png " ")

- **梯度的角度**
	边的方向与梯度方向垂直
$$\theta = tan^{-1} (\frac{\partial f}{\partial y}/\frac{\partial f}{\partial x})$$

- **梯度的模长（幅值）**
	可以说明是边缘的可能性大小
$$||\nabla f|| = \sqrt{(\frac{\partial f}{\partial x})^2+(\frac{\partial f}{\partial y})^2}$$

- 处理图像后：
	![](https://img-blog.csdnimg.cn/20200710114847529.png " ")
## 高斯滤波器
当图像的像素存在大量噪点时，相邻的像素差异大，所求梯度也会偏大，无法提取边缘信息。
![](https://img-blog.csdnimg.cn/2020071012030185.png " ")
**解决方案**

1. 平滑处理：使用平滑滤波器去噪，使图像信号变得平滑
2. 再对处理后的信号求导，取极值
![](https://img-blog.csdnimg.cn/20200710121354767.png " ")

3. 根据卷积的计算性质：$\frac{d}{dx}(f*g) = f * \frac{d}{dx}g$，先对平滑核求导，再进行卷积相乘来简化运算，减少运算量

![](https://img-blog.csdnimg.cn/20200710123012114.png " ")

 **高斯滤波器**
![](https://img-blog.csdnimg.cn/20200710125348951.png " ")

**高斯滤波器的导数**

参数选择的越小则保留的细节越多
 ![](https://img-blog.csdnimg.cn/20200710124831980.png " ")
## Canny 边缘检测
**门限化**

经过处理后，可以得到边缘图，但存在很多高频噪点，通过设置更高的门限，过滤噪点，使得到的边缘更“纯粹”

![](https://img-blog.csdnimg.cn/20200710161702718.png " ")

**非最大化抑制**

在通过高斯滤波后可以得到图像的大致轮廓线，由于图像的像素变换通常是缓慢改变的， 在处理后的图像中仍然存在大量的粗的“**边**”

![](https://img-blog.csdnimg.cn/2020071015573229.png " ")
**方案**
1. 检查像素是否沿梯度方向为局部最大值，选择沿边缘宽度的最大值作为边缘
![](https://img-blog.csdnimg.cn/2020071016033444.png " ")
2. 处理后
	![](https://img-blog.csdnimg.cn/20200710162723699.png " ")
	经过上面的处理后，已经可以较为粗糙的得到图像的边缘图，但仍然存在问题，在有些部分的边	缘不连续，失去了很多信息如上图的 `黄色区域` ，这是由于在门限化的过程中，设置过小，导致将需要的边缘滤除。
	
**双门限法**

1. 先使用高门限将较粗的边检测出来，这些边都是比较鲁棒的，是噪声的可能性极低
2. 再降低门限，将较细的边显现出来
3. 将与高门限过滤出的边连接的低门限边保留，滤除没有连接的（不连续的）噪声

![](https://img-blog.csdnimg.cn/20200710164509675.png " ")

4. 处理后可以得到更好的边缘效果

![](https://img-blog.csdnimg.cn/20200710164831228.png " ")

**学习资源：[北京邮电大学计算机视觉——鲁鹏](https://www.bilibili.com/video/BV1nz4y197Qv)**

