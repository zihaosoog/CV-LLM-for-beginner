## DirectPose: Direct End-to-End Multi-Person Pose Estimation
The regression-based method have the potential to detect very dense keypoints.

The heatmap-based task is only used as an auxiliary loss during training.
![Screenshot from 2022-10-26 16-39-38](https://user-images.githubusercontent.com/67272893/197977999-b29e71e6-800c-49c6-98e3-9943052be71b.png)
**KPAlign**

Locator predict the rough locations of the keypoints from high-level features with a larger receptive field.

Sampler samples feature acorrding to the above offsets from high-resolution low-level features with a smaller receptive field.

Predictor make the final keypoint predictions.
![Screenshot from 2022-10-26 16-57-45](https://user-images.githubusercontent.com/67272893/197982412-af2fc4c1-495b-47c3-9534-1fc0228f52d0.png)
**Heatmap Prediction**

Previous heatmap-based keypoint detection methods [1] generate unnormalized Gaussian distribution centered at each keypoint. we perform a per-pixel classification here for simplicity. Note that we make use of multiple binary classifiers (i.e., one-versus-all) and therefore the number of output channels is K instead of K + 1.

**GT of heatmap**

On the heatmaps, if a location is the nearest location to a keypoint with type t, the classification label for the location is set as t, 

where t ∈ {1, 2, ..., K}. Otherwise, the label is 0.
