# Top-down
1. Employ a heavy person detector
2. Single person pose estimation for each detection

# Bottom-up
1. Predict the heatmaps to detect all the keypoint
2. At the same time, group all the keypoint into individual persons
3. Postprocessing: pixel-level NMS, line integral, refinement, grouping.

# Multi-task
[1]. [YOLOPose](https://github.com/zh-song/Object-Detection-Papers/blob/Docments/Multi%20task/YOLO-Pose.md#yolo-pose)  
[2]. [DirectPose: Direct End-to-End Multi-Person Pose Estimation](https://github.com/zh-song/Object-Detection-Papers/blob/Docments/Multi%20task/DirectPose.md#directpose-direct-end-to-end-multi-person-pose-estimation)  
[3]. [Single-Stage Multi-Person Pose Machines](https://github.com/zh-song/Object-Detection-Papers/blob/Docments/Multi%20task/Single-Stage%20Multi-Person%20Pose%20Machines.md#single-stage-multi-person-pose-machines)
