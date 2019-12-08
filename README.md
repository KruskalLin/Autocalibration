# Simple Extrinsic Autocalibration Framework
In this work, we use unsupervised depth estimators to predict depth maps and optimize the distances between predicted depth maps and miscalibrated projected depth images. 

Estimator:

- Check out this branch Monodepth[1]

## Upsampling Toolbox

|  Method   | Linear | Nearest  | KNN | Barycentric | Grid Weight |
|  ----  | ---- | ----  | ----  | ----  | ----  |
| Result  | ![Linear](img/linear.png) | ![Nearest](img/nearest.png) | ![KNN](img/knn.png) | ![Barycentric](img/barycentric.png) |  ![Grid_weight](img/grid_weight.png) |


|  Method   | Anisotropic Diffusion[2] | SD-Filter[3]  | Geometry | Modified Spatial[4] | Total Generalized Variation |
|  ----  | ---- | ----  | ----  | ----  | ----  |
| Result  | ![Anisotropic](img/anisotropic.png) | ![SDFilter](img/sdfilter.png) | ![geometry](img/geometry.png) | ![Barycentric](img/spatial.png) |  ![tgv](img/tgv.png) |
