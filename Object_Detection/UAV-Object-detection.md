# Two major categories:
> Two stage

first generate a region proposal and then classify and locate candidate regions.

```markdown
# Detection head
Purpose: assign positive/negative samples
Out: classification and bbox
```

> One stage 

directly generate class probability and coordinate position (faster).

# Data processing

图像采集后预处理：

    1.增加训练样本数量
    2.增加样本尺寸和样本方向的多样性
    3.增加样本亮度

# Challenges

> image degradation

    Q: high-speed flight, camera rotation
    A: fuzzy, noisy
    P: noise reduction, camera distortion correction

> Object size

	Q: different altitudes (< 32x32 pixels)
	A: different size of objects

> Real-time

	Q: quick
	A: YOLO/SSD

> Uneven object intensity

	Q: 目标分布不均匀，overlap

# Strategy


> 分辨率/感受野

    1.特征金字塔
    2.空洞卷积/可变卷积

> 超分辨率

    pooling/anchor_imformation/crop

> 方向多样性/视角变化

	1.数据增强
	2.添加模块：多方向的RPN/anchor
	3.pooling/conv 提取方向鲁棒的特征

> 类别不均衡

	采样平衡/类别平衡

> 上下文语义信息