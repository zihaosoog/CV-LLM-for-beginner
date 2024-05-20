1. FPN 2个作用：**多尺度特征融合**  以及  **分而治之**

> 特点：

```
neck分为4种：单输入和多输入，单输出和多输出;
对于多输出模式中，只用C5特征作为输入和输入C3-C5多层输入效果不相上下
```

> 贡献

Dilated Encoder：代替FPN从C5中获得更大的感受野

<img src="/home/zhsong/Pictures/Screenshot from 2022-04-22 15-40-06.png" style="zoom:80%;" />

2. 样本均匀分配 Uniform Matching

> 特点

根据IoU确定正负样本中，**大GT box** 比小GT box更容易获得 **更多的正样本**

> 贡献

选择GT周围最近邻K个anchor作为正样本，这个k不会像ATSS动态IoU阈值选取，而是固定的。

ATSS 首先在 L 层特征图上为每个GT 选择 top k anchors，然后通过动态 IoU 阈值在 k × L anchors中采样正样本。