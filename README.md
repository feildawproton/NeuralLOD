# NeuralLOD
LOD with neural networks

I. Abstract
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
