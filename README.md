# DSEA-SMOTE
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

This paper uses the CWRU and SEU public datasets. If you use the code in this paper, please cite our paper and light up the star for our project. If you use it for commercial purposes, please contact the author of the paper.
