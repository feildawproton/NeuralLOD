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
