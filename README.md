# Bias-conflict Sample Synthesis and Adversarial Removal Debias Strategy for Temporal Sentence Grounding in Video

Code for AAAI 2024 paper "Bias-conflict Sample Synthesis and Adversarial Removal Debias Strategy for Temporal Sentence Grounding in Video" [arxiv].

## Installation

We provide the environment file for anaconda.

You can build the conda environment simply by

```
conda env create -f environment.yml
```

## Dataset Preparation

#### Features and Pretrained Models

You can download the prepared visual features from [Box Drive](https://app.box.com/s/h0sxa5klco6qve5ahnz50ly2nksmuedw), In the provided link, The video features extracted by C3D are stored in a folder named after org, while the folder named after new stores the video features extracted by I3D, you place them to the `./data/features` directory. Such as you can place /features/charades/new as /features/charades/i3d.

The two partitioning methods for the characters dataset have been provided here. The original partitioning is located in data/dataset/characters/org, while the heterogeneous partitioning is located in data/dataset/characters/cd. The activitynet dataset is similar to it.

#### Word Embeddings

Download the word embeddings from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip) and place it to `./data/dataset` directory.

## Quick Start

If you want to use features which was extracted by C3D, please change video_feature_dim as 4096. 

The pre-trained model weigths are provuded in [Baidu](https://pan.baidu.com/s/1V35SfR9etbq38xJzEAehGQ) passward: 'az0j' or [Google Drive](https://drive.google.com/drive/folders/1hyO-bj1e3dS6R3bwnTgcLC6drb63kvV6)

### Charades-CD

baseline:

```
python train_cd.py --task charades --video_feature_dim 1024 --epochs 50 --init_lr 0.001 --verbose 2 --vf org --qf cd --model_name base_none_base
```

debias model:

```
python train_cd.py --task charades --video_feature_dim 1024 --epochs 50 --init_lr 0.001 --verbose 2 --vf org --qf cd --model_name add2vaqfe_vqc128_vqc
```

### ActivityNet-CD

baseline:

```
python train_cd.py --task activitynet --video_feature_dim 1024 --epochs 50 --init_lr 0.001 --verbose 2 --vf org --qf cd --model_name base_none_base
```

debias model:

```
python train_cd.py --task activitynet --video_feature_dim 1024 --epochs 50 --init_lr 0.001 --verbose 2 --vf org --qf cd --model_name add2vaqfe_vqc128_vqc
```

## Citation

Please cite our papers if you find them useful for your research.

```
  author    = {Zhaobo and Qi, Yibo and Yuan, Xiaowen and Ruan, ShuHui and Wang, Weigang and Zhang, QingMing and Huan},
  title     = {Bias-conflict Sample Synthesis and Adversarial Removal Debias Strategy for Temporal Sentence Grounding in Video},
  booktitle = {AAAI},
  year      = {2024},
}
```
