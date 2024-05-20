# GhostNet: More Features from Cheap Operations

## <b>Intoduction</b>:
For mobile devices : network pruning(网络剪枝), low-bit quantization(低位量化), knowledge distillation(知识蒸馏)

## <b>Approach</b>:
利用 CNN 生成的中间特征图之间的相关性以及冗余，用 ghost 减少计算量，生成更多特征图

![fig2](https://pic4.zhimg.com/80/v2-6cf3c59130c5b6c50dac8bf19e68e35f_720w.jpg)

计算思路类似于深度可分离卷积或者1
×1卷积，将特征图生成分为两个步骤:

1. 首先通过减少卷积核（conv 即普通卷积）生成输出通道为 m 的特征，而 a 卷积输出通道为 n
2. 通过cheap operation，利用第 i 通道特征生成第 j 通道特征，一个 i 特征对应 s-1 个生成的 j 特征以及1个直接通道复制的特征，m 个 i 特征就对应 m×s 个 j 特征，而 m×s=n（ s 即扩展倍数）
   
![fig3](https://pic4.zhimg.com/80/v2-d3ea73227ffafd43b51723fc6a08c903_720w.png)

<b>FLOPs</b>:
> FLOPs = input_channel × output_channel × kernel_size × output_feature_map_size

> parameres = input_channel × output_channel × kernel_size

![979id4hyco](https://user-images.githubusercontent.com/67272893/149614718-5ada63dd-d526-4c70-9eb7-3a9c1e2f0301.png)

<b>深度可分离卷积（DWC）</b>：

![1489774-20200823105310479-506198517](https://user-images.githubusercontent.com/67272893/149614676-0142691b-5df5-4a56-b29d-1bf4388bad8e.png)

Depthwise Convolution：

![1489774-20200823105331585-1644804158](https://user-images.githubusercontent.com/67272893/149614703-779c1971-de96-4e8b-bed1-fb0ea1e1a31a.png)

Pointwise Convolution：实际为1×1卷积，在DWC中它起两方面的作用。
1. 让DWC能够自由改变输出通道的数量；
2. 对Depthwise Convolution输出的feature map进行通道融合

![1489774-20200823105348729-1761867796](https://user-images.githubusercontent.com/67272893/149614707-64bae681-2eea-4996-ac38-c17c1448d254.png)

<b>Ghost Bottlenecks</b>:

类似与residual block，其中采用了DWC

![fig5](https://pic4.zhimg.com/80/v2-23f8cbea9094dfde878f403092c8d103_720w.jpg)




## <b>References</b>:
> [作者解读](https://zhuanlan.zhihu.com/p/109325275)

> [Ghost笔记](https://cloud.tencent.com/developer/article/1745462)

> [深度可分离卷积](https://zhuanlan.zhihu.com/p/166736637)

> [部分代码](https://zhuanlan.zhihu.com/p/148856494)

> [实验分享](https://zhuanlan.zhihu.com/p/115844245)
