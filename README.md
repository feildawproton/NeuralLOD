# NeuralLOD
LOD with neural networks

## Abstract
Levels-of-detail (LODs) are an important concept in real-time graphics where higher detail models, with
higher memory costs, replace lower detail models upon closer inspection. Recently, LOD has been made
hierarchical so that memory and calculations from lower LODs are reused and the memory and
computation required by the higher LOD is reduced. Additionally, the decision surface of neural
networks have been used as a natural shape representation for learning tasks. In this research we
introduce the idea of using a single neural network to produce hierarchical LODs. This is accomplished
with multiple outputs connected to different depths of the network, and learning an occupancy function
that takes a coordinate and returns whether the shape is there or not. The result is a network that can
be loaded and evaluated layer wise, where the later layers reuse the output of lower detail layers. This
research demonstrate this concept working in 2 dimension.

## Introduction
Triangle meshes are the most common 3D representation for real-time applications. They are
also quite common in artistic and creative applications. Arbitrary triangle meshes require arbitrary
number of vertices and arbitrary topology. Therefore learning tasks with meshes typically use templates
that can be deformed with latent control parameters. Voxels are the most common 3D representation
for learning tasks. This is because voxels are the 3D equivalent of pixels. The main drawback of voxels is
the relatively large amount of memory they occupy. Recently, the decision surface of neural networks
have been used as a 3D occupancy function. This is a straightforward representation for learning
tasks.
Multiple Levels-of-detail (LODs) are often required by real-time application, to reduce the total
amount of memory occupied and computations performed at any given moment. However, there is
overhead when unloading unused LODs and there is data duplication in the similarity between model
LODs. Recent work has used hierarchical LOD, where higher levels depend on the lower. This
reduces the size of, and computations required to evaluate, the higher LOD models. Since the lower
LOD model is reused there is not overhead from unloading them. This research combines neural
occupancy functions with hierarchical levels of detail. This approach could be useful for learning and
real-time tasks

## Method, Experiments, and Results
Our goal is to design a neural network so that its decision surface is useful for rendering at
different level of detail. For this to be of any utility, it must be possible to load only part of model for
evaluation. Ideally, loading more detail should not require replacing the parts of the model already
loaded into memory. It should be hierarchical, such as the mesh data in state-of-the-art implementations. To this end we design a
neural network with multiple outputs connected to different hidden depths.
![architectures](https://user-images.githubusercontent.com/56926839/161315054-0b202e66-cd93-49ba-9bce-7bb7f685f529.png)
Figure 1: Graphs of 3 designs. From left to right: no shared output weights, shared output weights, and shared output weights
with multiclass input.

This design requires that earlier layers learn to create useful representations for later layers and
for their own output layers. In figure 1, the design on the left uses different output weights for each
level of detail. These output weights will not be used by later layers and are a waste of data when
higher detail is required.
To address this we tested sharing weights across the output layers. This reduces the memory
footprint of the entire model and eliminates wasted portions of the model. All layers of the model
previously loaded will be reused when higher LOD is required.
Dropout, with a low rate of 0.1, is used to aid in convergence. All hidden layers have the same
width. In all cases, the neural network acts as an occupancy map. It takes in coordinates (in this case
2D) and returns whether or not the model is there:

f(x,y)→[0,1].

The returned values can be anywhere between 0 and 1. It is up to the renderer how to treat this
value. Treating it as a probability coincides well with the meaning of the training data.
Because there are multiple outputs, there will be multiple loss functions (one for each output).
The final loss function will be a weighted sum of these loses. Error can be backpropagated as normal.
Mean squared error was used for all losses.

loss= ∑▒〖w_i loss_i 〗

For simplicity and to prove the concept, these experiments were conducted on 2D images.  The images were 64x64 black and white png’s made for this experiment.  Each image is treated as a dataset containing 4096 samples with input (x,y) and labels [0,1].
![image](https://user-images.githubusercontent.com/56926839/161315572-8520b325-f457-4838-88e5-3b74240bdbab.png)
Figure 2:  The five images used in these experiments.  From left to right:  Leto II, Big Brother, Soma, Simon, and tanstaafl.

Training was conducted with batch size equal to that of the size of the images (4096) and run for 32,768 epochs.  The Adam optimizer was used with a learning rate of 0.001.  In order to establish a baseline of comparison, a fully connected neural network with the same architecture as the ones in Figure 1, without multiple outputs, was trained to recreate these images.
![image](https://user-images.githubusercontent.com/56926839/161315649-6ffc2476-d4dd-4990-927f-2e5aca5cb862.png)
Figure 3:  Reconstructed images.  They are at a higher resolution that the source data to demonstrate that NNs can be super-sampled.





