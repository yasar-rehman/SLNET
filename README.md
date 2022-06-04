# SLNET
Stereo Face Liveness Detection using Convolutional Neural Networks

The paper can be found at: http://www.ee.cityu.edu.hk/~lmpo/publications/2019_ESA_SLNet.pdf

# Abstract
Current state-of-the-art dual camera-based face liveness detection methods utilize either hand-crafted features, such as disparity, or deep texture features to classify a live face and face Presentation Attack (PA). However, these approaches limit the effectiveness of classifiers, particularly deep Convolutional Neural Networks (CNN) to unknown face PA in adverse scenarios. In contrast to these approaches, in this paper, we show that supervising a deep CNN classifier by learning disparity features using the existing CNN layers improves the performance and robustness of CNN to unknown types of face PA. For this purpose, we propose to supervise a CNN classifier by introducing a disparity layer within CNN to learn the dynamic disparity-maps. Subsequently, the rest of the convolutional layers, following the disparity layer, in the CNN are supervised using the learned dynamic disparity-maps for face liveness detection. We further propose a new video-based stereo face anti-spoofing database with various face PA and different imaging qualities. Experiments on the proposed stereo face anti-spoofing database are performed using various test case scenarios. The experimental results indicate that our proposed system shows promising performance and has good generalization ability.

# Citation
If you find this code useful, please cite the following article:

```
@article{rehman2020slnet,
  title={SLNet: Stereo face liveness detection via dynamic disparity-maps and convolutional neural network},
  author={Rehman, Yasar Abbas Ur and Po, Lai-Man and Liu, Mengyang},
  journal={Expert Systems with Applications},
  volume={142},
  pages={113002},
  year={2020},
  publisher={Elsevier}
}
```
