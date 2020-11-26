# BalaGAN: Image Translation Between Imbalanced Domains via Cross-Modal Transfer
###<a href="https://orpatashnik.github.io/BalaGAN/">Project Page</a> | <a href="https://arxiv.org/abs/2010.02036">Paper</a> | <a href="https://www.youtube.com/watch?v=yNBmY5M8GvE">Video</a>

<img src="docs/teaser.gif" width=512 alt="balagan-teaser">
<img src="docs/img/balagan.png" width=512 alt="balagan">



> State-of-the-art image-to-image translation methods tend to struggle in an imbalanced domain setting, where one image domain lacks richness and diversity. 
>We introduce a new unsupervised translation network, BalaGAN, specifically designed to tackle the domain imbalance problem. 
>We leverage the latent modalities of the richer domain to turn the image-to-image translation problem, between two imbalanced domains, into a balanced, multi-class, and conditional translation problem, more resembling the style transfer setting. 
>Specifically, we analyze the source domain and learn a decomposition of it into a set of latent modes or classes, without any supervision. 
>This leaves us with a multitude of balanced cross-domain translation tasks, between all pairs of classes, including the target domain. 
>During inference, the trained network takes as input a source image, as well as a reference or style image from one of the modes as a condition, and produces an image which resembles the source on the pixel-wise level, but shares the same mode as the reference. 
>We show that employing modalities within the dataset improves the quality of the translated images, and that BalaGAN outperforms strong baselines of both unconditioned and style-transfer-based image-to-image translation methods, in terms of image quality and diversity.


## Prerequisites
- Linux (may work on windows and macOS but was not tested)
- cuda 10.1
- Anaconda3
- pytorch (tested on >=1.5.0)
- tensorboardX
- faiss-gpu
- opencv-python

## Training

#### Data Preparation
A dataset directory should have the following structure:

```
dataset
├── train
│   ├── A
│   └── B
└── test
    ├── A
    └── B
```
where A is the source domain, and B is the target domain.

#### Train
The main training script is `train.py`. It receives several command line arguments, for more details please the file.
The most important argument is a path to a config file. An example for such a file is provided in `configs/dog2wolf.yaml`

#### Tracking The Training
For each experiment, a dedicated directory is created, and all the outputs are saved there.
An experiment directory contains the following:
- **logs** directory with a tensorboard file which contains the losses along the training, and images produced by the model.
- **images** directory, in which the images are saved as files.
- **checkpoints** directory in which checkpoints are saved along the training.

We highly recommend using [trains](https://github.com/allegroai/trains) to track experiments! 

#### Resume An Experiment
To resume an experiment, provide the `--resume` flag to the main training script. 
When providing this flag, the state of the latest experiment with the same `--exp_name` is loaded.

#### Pretrained Models
Coming soon...

## Citation
If you use this code for your research, please cite our paper
[BalaGAN: Image Translation Between Imbalanced Domains via Cross-Modal Transfer](https://arxiv.org/abs/2010.02036)
```
@article{patashnik2020balagan,
      title={BalaGAN: Image Translation Between Imbalanced Domains via Cross-Modal Transfer}, 
      author={Or Patashnik and Dov Danon and Hao Zhang and Daniel Cohen-Or},
      journal={arXiv preprint arXiv:2010.02036},
      year={2020}
}
```





