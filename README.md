# EEG_Shape_Reconstruction
## Title: Multi-task Generative Adversarial Learning on Geometrical Shape Reconstruction from EEG Brain Signals 

**PDF: [ICONIP2019](http://ajiips.com.au/papers/V15.2/v15n2_40-47.pdf), [arXiv](https://arxiv.org/abs/1907.13351)**

**Authors: [Xiang Zhang](http://xiangzhang.info/) (xiang_zhang@hms.harvard.edu), Xiaocong Chen, Manqing Dong, Huan Liu, Chang Ge, [Lina Yao](https://www.linayao.com/) (lina.yao@unsw.edu.au)**

## Overview
This repository contains reproducible codes for the proposed EEG_Shape_Reconstruction model.
In this work, we propose a novel approach to reconstruct the geometrical shape based on the brain signals. We first develop a framework learning the latent discriminative representation of the raw EEG signals, and then, based on the learned representation, we propose an adversarial reconstruction framework to recover the geometric shapes which are visualizing by the human. In particular, we propose a semantic alignment method to improve the realism of the generated samples and force the framework to generate more realistic geometric shapes. The proposed approach is evaluated over a local dataset and the experiments show that our model outperforms the competitive state-of-the-art methods both quantitatively and qualitatively. Please check our paper for more details on the algorithm.

<p align="center">
<img src="https://raw.githubusercontent.com/xiangzhang1015/EEG_Shape_Reconstruction/master/Demonstration%20of%20the%20qualitative%20comparison.PNG", width="600", align="center", title="Demonstration of the qualitative comparison. Our model can reconstruct all the shapes correctly which have the highest similarity with the ground truth.">
</p>
<p align = "center">
<b>  Qualitative comparison. Our model can reconstruct all the shapes correctly which have the highest similarity with the ground truth.</b>
</p>


## Code
**EEG feature learning**

In [EEG_featurelearning.py](https://github.com/xiangzhang1015/EEG_Shape_Reconstruction/blob/master/EEG_featurelearning.py), we put our CNN classifier which used to extract EEG features.

**Multi-task GAN**

In [GAN_shape.py](https://github.com/xiangzhang1015/EEG_Shape_Reconstruction/blob/master/GAN_shape.py), we build a multi-task GAN along with the semantic alignment to reconstruct the shapes.

## Citing
If you find our work useful for your research, please consider citing this paper:

    @article{zhang2019multi,
      title={Multi-task Generative Adversarial Learning on Geometrical Shape Reconstruction from EEG Brain Signals},
      author={Zhang, Xiang and Chen, Xiaocong and Dong, Manqing and Liu, Huan and Ge, Chang and Yao, Lina},
      journal={arXiv preprint arXiv:1907.13351},
      year={2019}
    }


## Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <xiang.alan.zhang@gmail.com>.


## License

This repository is licensed under the MIT License.
