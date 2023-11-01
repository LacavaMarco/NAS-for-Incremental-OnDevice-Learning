# Neural Architecture Search for Tiny Incremental On-Device Learning

This repository contains the code developed for my Master's Thesis at Politecnico di Milano under the supervision of Prof. Manuel Roveri.

Incremental on-device learning is a promising research field in TinyML which consists in the ability of models to adapt to new data without constant reliance on cloud connectivity. TyBox is a toolbox for the automatic design and code-generation of incremental on-device TinyML models that addresses this challenge effectively. 

In parallel, Neural Architecture Search (NAS) is a powerful AutoML approach which aims to automate the design of a neural architecture optimized on its accuracy and computational requirements for a given task and dataset. Constrained NAS (CNAS) expands on the concept of Neural Architecture Search, incorporating TinyML-inspired constraints to strike a balance between performance and resource efficiency. However, it primarily targets mobile devices with greater computational resources, making it unsuitable for MCUs. 

The literature contains solutions that implements NAS for TinyML, while no incremental on-device learning NAS solutions are currently available in the literature. 
**The aim of this thesis is to provide a framework for the automatic design and deployment of incremental on-device models. The proposed framework bridges the gap between search and deployment: starting from an architecture discovered by CNAS, it generates an incremental on-device version of the architecture that meets the severe requirements of tiny devices and provides the corresponding file for the deployment on device.**
Performance tests on multi-image classification tasks demonstrate the effectiveness of the proposed methodology, showcasing the competitiveness of the compressed incremental models compared to the original static CNAS models and alternative incremental toolboxes.


## Background

### CNAS
[CNAS](https://github.com/matteogambella/NAS) is a bi-objective algorithm that extends NAS by imposing constraints related to the requirements of the device considered for the deployment. 
The two metrics it optimizes are the model accuracy and $Φ_{CNAS}$,  which accounts jointly for the number of parameters, MACs and activations of a designed architecture. 
**The inclusion of constraints allows the design of models that can operate on devices with limited resources. However, the models produced by CNAS predominantly target mobile devices, which have higher computational resources compared to MCUs, so this technique is not suitable for deployment on tiny devices.**

### TyBox
[TyBox](https://github.com/pavmassimo/TyBox) is a state-of-the-art toolbox for the automatic design and code-generation of incremental on-device TinyML models. It takes as input a static model and a memory constraint and design an incremental version of the static input model that meets the imposed constraints and the code and library necessary for deployment on device.
The incremental module has the task of designing the incremental model, composed by a fixed feature extraction block, an incrementally learnable classification block and a buffer B. The purpose of the buffer B is to store supervised samples acquired from the field to retrain the incremental model every time new data becomes available. 
The code-generation module receives in input the incremental model and generates files required for the deployment on device.
**While TyBox aims at automatizing the deployment of incremental models, it still requires as input an expert design ML model.**

## Proposed framework

![Proposed framework image](https://github.com/LacavaMarco/NAS-for-Incremental-OnDevice-Learning/blob/main/Proposed_framework/Proposed_framework.png)

### Structured pruning
The smallest optimal CNAS model has a size that amounts to 8.72MB, a dimension which significantly exceeds the operational capabilities of MCUs. To achieve a reduction in model size, structured pruning has been selected as the first approach. 
In contrast to conventional pruning techniques, which simply convert the weights with smaller magnitudes to zeros, structured pruning removes connection between the layers of a model. In our scenario, structured pruning has been applied to the Convolutional layers to eliminate their least relevant filters. The filter’s relevance is estimated by computing the L1-norm of the filter’s weights. 

### Full-integer quantization
Pruning alone proves insufficient in achieving a network that meets the restrictions of tiny devices. Consequently, the TyBox toolbox has been extended to incorporate full-integer quantization. Full-integer quantization has been identified as the optimal method in this scenario due to its capability to convert all the model components into 8-bit integer data, compared to other quantization techniques which may leave some amount of data in floating-point.
The primary advantage of a TyBox incremental model lies in its incrementally learnable classifier. In order to reduce the incremental model dimension while maintaining its prediction abilities, quantization is exclusively applied to the static feature extractor. The incremental classifier retains a 32-bit resolution, ensuring a more precise classification process is maintained. To match the different resolutions between the feature extractor and the classifier, the features undergo a dequantization process before being passed to the classifier. The same process is also employed to the quantized samples stored in the buffer before being utilized for re-training the classifier.
Once the tiny incremental on-device version of the CNAS model has been produced, the framework also provides the cpp codes and libraries for the deployment on device.

## Applications scenarios
To validate our solution, we decided to analyze different application scenarios: concept drift, incremental learning, and transfer learning.
The experimental setting concerns the image classification on a multi-class problem. For this purpose, CIFAR-10 and [Imagenette](https://github.com/fastai/imagenette) (10 classes color datasets) have been considered. 
Two distinct solutions are considered for the comparison: an incremental version of the baseline model, and an incremental model designed with TinyOL, an alternative incremental on-device toolbox, which rather than using a buffer performs the incremental training only on the latest supervised sample received.

## Structure of the repository
- [Proposed_framework](https://github.com/LacavaMarco/NAS-for-Incremental-OnDevice-Learning/tree/main/Proposed_framework) contains the code for **structured pruning**, while the code for the **implementation of quantization in TyBox** can be found [here](https://github.com/pavmassimo/TyBox/tree/feature-extractor-quantization).
- [Experiment_notebooks](https://github.com/LacavaMarco/NAS-for-Incremental-OnDevice-Learning/tree/main/Experiment_notebooks) contains the notebooks of the performed experiments.
- A more in depth description of the work can be found in the [Executive Summary](https://github.com/LacavaMarco/NAS-for-Incremental-OnDevice-Learning/blob/main/Executive%20Summary%20Lacava.pdf) and [Master Thesis](https://github.com/LacavaMarco/NAS-for-Incremental-OnDevice-Learning/blob/main/Master%20Thesis%20Lacava.pdf) documents.
