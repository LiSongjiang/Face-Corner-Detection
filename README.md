# Face-Corner-Detection

##Intro.

Detect eye corners, pupil center, nostrils and mouth corners at low cost.

使用`OpenCV`，快速检测**正脸**上的双眼**眼角**，**瞳孔中心**，**鼻孔**和**嘴角**。

##Files

包含了**4**个文件，分别是

1. `corners_detection.cpp` 
2. `corners_detection.h`
3. `haarcascade_frontalface_alt.xml`
4. `haarcascade_mcs_eyepair_big.xml`

其中1和2是代码文件，3和4是`OpenCV`自带的`HaarCascade`人脸和双眼检测文件。使用前需要先运行`initializeModule(String CASCADE_FILENAME_FACE, String CASCADE_FILENAME_EYEPAIR)`导入两个文件。

##Info.

Version 1.0

2015-9-6

MrZigZag @ Peking University

Email: lisongjiang@pku.edu.cn

