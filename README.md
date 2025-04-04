# DSEA-SMOTE

## Code background
In industrial production, the imbalance between sparse failure signals and abundant normal signals
biases fault diagnosis models towards the normal class,
reducing accuracy and reliability. Existing synthetic data
methods help mitigate this but often neglect spatiotemporal long-distance dependencies and local feature similarities, limiting their effectiveness. To address these limitations, we propose Deep Squeeze-enhanced Axial Transformer Synthetic Minority Oversampling Technique (DSEA-SMOTE). This method integrates a specially designed
continuous wavelet transform data filtering preprocessing
technology module that converts one-dimensional time-domain data into a two-dimensional feature map, enhancing model performance while simplifying feature learning.
It then captures spatiotemporal long-distance dependencies and feature similarities in space-time slices through
the novel Squeeze-enhanced Axial Attention mechanism
and Auxiliary Feature Classifier. A Multi-category Sample
Feature Filtering Technology module is also introduced to
further improve synthesis quality. Additionally, we refine
the loss function based on the Auxiliary Feature Classifier
to enhance generation quality. Experimental evaluations
on two real-world datasets show that DSEA-SMOTE outperforms recent techniques. Ablation experiments further
verify the effectiveness of each component in our design.Future work will extend DSEA-SMOTE to medical image
synthesis for rare diseases, remote sensing image synthesis for
extreme natural disasters, radar signal synthesis for aviation
objects, and explore its use with other modalities.

## Code repository introduction
This code repository is the source code of the paper "Deep Squeezze-enhanced Axial Transformer SMOTE: A Novel Approach For Imbalance-bearing Fault Diagnosis". It mainly provides the framework code and related processing code of the novel data synthesis model framework DSEA-SMOTE. It also provides related data files.Below is a schematic diagram of the model framework of the entire paper.
![Example Image](README/README-1.jpg)
Below are the results of the model synthesizing samples under extremely unbalanced conditions (BR 1:400) (from left to right: original sample, synthesized sample color map, synthesized sample grayscale map).
![Example Image](README/README-2.bmp)
![Example Image](README/README-3.bmp)
![Example Image](README/README-4.bmp)
![Example Image](README/README-5.bmp)
Meanwhile, the following is an explanation for the long distance dependence of sample synthesis in space-time and the similarity of overall and local features of space-time slices.
![Example Image](README/README-7.bmp)
Experimental hardware setup and software setup:
Software Environment PyTorch 2.3.0, Python 3.12 (Ubuntu 22.04), Cuda 12.1
CPU 12 vCPU Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz
GPU RTX 3080 Ti (12GB)
Memory 90GB

## Usage Statement
This paper uses the CWRU and SEU public datasets. If you use the code in this paper, please cite our paper and light up the star for our project. If you use it for commercial purposes, please contact the author of the paper.

## Citation format
The citation format of the paper repository is：
@software{,
  author       = {Hongliang Dai, Dongjie Lin, Junpu He, Xinyu Fang, Siting Huang},
  title        = {amstlldj/DSEA-SMOTE: v1.0.0},
  month        = apr,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.15148882},
  url          = {https://doi.org/10.5281/zenodo.15148882},
  swhid        = {swh:1:dir:0bb059f1238a40321849a1b562115347aed38a88
                   ;origin=https://doi.org/10.5281/zenodo.15148881;vi
                   sit=swh:1:snp:61d7b3083fcc7fd599982eb07fd885fa0a42
                   a551;anchor=swh:1:rel:66dc4daa4cd603c2933ddcae1fc0
                   b9e43c864b6b;path=amstlldj-DSEA-SMOTE-5a1079d
                  },
}

or

@misc{DSEA-SMOTE, author = {Hongliang Dai, Dongjie Lin, Junpu He, Xinyu Fang, Siting Huang}, title = {DSEA-SMOTE}, year = {2024}, publisher = {Zenodo}, doi = {10.5281/zenodo.15148882}, url = {https://github.com/amstlldj/DSEA-SMOTE}

## Subsequent maintenance plan
There is still room for improvement in the readability of the project code. We plan to refactor the code of the entire project and upload the weight file in the future.(2025-3-26)
If you want to see our latest work, you can take a look at our SCQ-CFGRF model project, which is more mature and has uploaded the model's weight file, which can be run directly for testing.(https://github.com/amstlldj/SCQ-CFGRF)
