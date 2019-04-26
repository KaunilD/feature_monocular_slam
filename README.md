## Dependencies:
1. [g2opy](https://github.com/uoip/g2opy) for Pose Graph Optimization
2. [pygolin](https://github.com/stevenlovegrove/Pangolin) for rendering pointcloud and keyframes using OpenGL
3. python ~> 3.5.6
4. numpy - pip install numpy
5. [OpenCV3] (https://docs.opencv.org/3.0-beta/doc/tutorials/introduction/linux_install/linux_install.html) build opencv from source.


## Usage:
```
python main.py
```
Let the slam system run for a few frames or 5 minutes. The posegraph and pointcloud will be rendered in the pangolin viewer.
![posegraph](https://github.com/KaunilD/feature_monocular_slam/blob/master/res/screen.png)

