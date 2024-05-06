# FairGen
### Code for the paper: [FairGen: Towards Fair Graph Generation](https://arxiv.org/html/2303.17743v3)

# Introduction
FairGen is an end-to-end deep generative model that can directly learn from the raw graph while preserving the fair graph structure of both the majority and minority groups. It jointly trains a label-informed graph generation module and a fair representation learning module by progressively learning the behaviors of the protected and unprotected groups, from the `easy' concepts to the `hard' ones. In addition, it incorporates a generic context sampling strategy for graph generative models, which is proven to be capable of fairly capturing the contextual information of each group with a high probability. 


### Requirement:
* Scipy < 1.13 (tested on 1.12)

### Tested Environment:
* Python: 3.12
* Pytorch: 2.3 
* Cuda: 12.1
* Scipy: 1.12
* Scikit-learn: 4.3.2
* Gensim: 1.4.2

### Environment and Installation:
1. conda env create -f environment.yml
2. conda activate fairgen

### Command
1. Unzip the dataset:
```
unzip data.zip
```

2. Training and evaluation:
```
python main.py -d FLICKR -b
```

### Some important parameters:
* -d: the name of the dataset
* -g: the index of the gpu, 0 is the default value. If not using gpu, ignore this flag.
* -b: the biased random walk or unbiased random walk. Biased random walk depends on the node proximity, while unbiased random walk is independent of node proximity. The default value is a biased random walk
 with this flag.

### Evaluation：
The edge list of the synthetic graph is stored in the directory: "'./data/FLICKR/FLICKR_output_edgelist_0_2.txt'".

The final results will be stored in the directory: "'./data/FLICKR/FLICKR_output_edgelist_0_2_metric.txt".


### Please cite our paper if you find it useful:
@article{zheng2023fairgen,

  title={Fairgen: Towards fair graph generation},
  
  author={Zheng, Lecheng and Zhou, Dawei and Tong, Hanghang and Xu, Jiejun and Zhu, Yada and He, Jingrui},
  
  journal={arXiv preprint arXiv:2303.17743},
  
  year={2023}
}
