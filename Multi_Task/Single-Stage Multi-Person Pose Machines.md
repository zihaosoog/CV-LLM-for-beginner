## Single-Stage Multi-Person Pose Machines
**Structured pose representation**
1. predict displacements between body joints and the root joint.
2. we exploit the person centroid as the root joint of the person instance.

**Hierarchical SPR**

we divide the root joint and body joints into four hierarchies based on articulated kinematics [20] by their degrees of freedom and extent of deformation.

**GT of root joint**: 高斯heatmap

**GT of displacements**: Root Joint 為中心, τ為半徑範圍內的點到 body joints 的位移向量
