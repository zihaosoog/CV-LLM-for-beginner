## YOLO-Pose
For each anchor, the box head predicts the $(x,y,w,h,class score)$, the keypoint head predicts the 17 keypoints of person.

Each keypoint includes $(x,y,conf)$. 

Ground truth of keypoint conf: If a keypoint is either visible or occluded, then the ground truth confidence is set to 1 else if it is outside the field of view, confidence is set to zero.

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/67272893/197978288-02aeac17-d291-422a-8fc5-e031cd1b63ba.png">
</p>

The loss of keypoint consists of two parts: the loss of $(x,y)$ and loss of $conf$.

The loss of $(x,y)$. Hence, if a ground truth bounding box is matched with $k_{th}$ anchor at location $(i,j)$ and scale $s$, we predict the keypoints with respect to the center of the anchor. 

The loss of $conf$, visibility flags for keypoints are used as ground truth.

<p align="center">
  <img src="https://user-images.githubusercontent.com/67272893/197978359-e8c73af9-95e4-4f9d-b710-2ef449e7a036.png">
</p>

![Screenshot from 2022-10-26 16-42-59](https://user-images.githubusercontent.com/67272893/197978695-cceee35a-c65a-4c16-9379-c6ef826513dd.png)
