# NeuralLOD
LOD with neural networks. 2021 for ECE8990.

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

Table 1: Reference Reconstructions
Image	#   Hidden	Width	  Final MSE
tanstaafl	  4	    256	    0.0069
Leto II	    4	    256	    0.00098193
Simon	      3	    256	    0.002
Big Brother	3	    256	    0.00028935
Soma	      3	    128	    0.0002559

Figure 4 shows the LOD effect we are trying to achieve.  The images on the left were generated by a neural network trained in the traditional manner to reconstruct the image on the final output layer.  The intermediate layers were visualized to show what the network was learning.  The intermediate layers learn representations useful to the final layers and these representations are not necessarily useful for visualization.  The images on the right were produced by a network whose loss function included reconstruction error for output layers connected to each hidden layer.  This is a visualization of the desired behavior.  The accuracy of the representation increases as more hidden layers are loaded but the early representations are still decent.

![image](https://user-images.githubusercontent.com/56926839/161316193-68da41ba-54f4-4def-9a6d-28ffb68d6034.png)
Figure 4:  Demonstration of the desired outcome.  The images on the left represent reconstruction with no weight assigned to the quality of the intermediate LODs.  The reconstruction on the right uses equal loss weights.

The same number of hidden layers and width were used in the following experiments for their respective images: Compare final LOD reconstruction vs reference reconstruction, shared output weights vs multiple output weights, and equal vs unequal weighted output losses.

![image](https://user-images.githubusercontent.com/56926839/161316257-48003c75-1814-4b90-aa31-07da1ced3487.png)
Figure 5:  Examples of hierarchical reconstructions.  From left to right: Output weights not shared and loss weights equal for Big Brother; output weights not shared and loss weights unequal for tanstaafl; shared output weights and equal loss weights for Soma; share output weights and unequal loss weights for Leto II.

Table 2: Reconstruction Error for various Images and Hyperparameters
Image	Hidden Layers	Width	Output	Loss Weights	Final MSE @ Output Layer
					1	2	3	4
Soma	3	128	Not Shared	(1,1,1)	0.0245	0.00073692	0.00018707	n/a
				(.25,1,16)	0.0213	0.00084567	0.0001308	n/a
			Shared	(1,1,1)	0.0261	0.0013	0.00031969	n/a
				(.25,1,16)	0.0278	0.00079919	0.00016157	n/a
Big Brother	3	256	Not Shared	(1,1,1)	0.0218	0.0013	0.0003163	n/a
				(.25,1,16)	0.0253	0.0012	0.00028436	n/a
			Shared	(1,1,1)	0.0238	0.0022	0.00058989	n/a
				(.25,1,16)	0.035	0.0027	0.00039654	n/a
Simon	3	256	Not Shared	(1,1,1)	0.0606	0.006	0.0023	n/a
				(.25,1,16)	0.0758	0.0089	0.002	n/a
			Shared	(1,1,1)	0.0648	0.0126	0.0039	n/a
				(0,0,1)	0.3209	0.261	0.0034	n/a
				(.25,1,16)	0.1097	0.0238	0.0034	n/a
Leto II	4	256	Not Shared	(1,1,1,1)	0.0184	0.0027	0.0012	0.00096177
				(.25,1,4,16)	0.0198	0.0039	0.0013	0.00090129
			Shared	(1,1,1,1)	0.0202	0.0046	0.022	0.0017
				(.25,1,4,16)	0.0248	0.0071	0.0025	0.0017
tanstaafl	4	256	Not Shared	(1,1,1,1)	0.0667	0.0136	0.008	0.007
				(.25,1,4,16)	0.0064	0.0171	0.0089	0.0064
			Shared	(1,1,1,1)	0.0649	0.0151	0.011	0.01
				(1,1,1,100)	0.0833	0.0337	0.0182	0.0098
				(.25,1,4,16)	0.0826	0.0206	0.0115	0.0089


Comparing the results from Tables 1 and 2, and looking at the reconstructed images, we can see that the level-of-detail networks are able to produce a good final reconstruction.  They are also able to learn and produce lower detail reconstructions.  Sharing network weights across the output layers did reduce the accuracy of the reconstructions compared to not sharing weights.  Weighting the loss functions had a minor impact on reconstruction accuracy.  Most notably it reduced the accuracy of the low level reconstruction, while providing a barely noticeable increase in reconstruction accuracy of the final output. 

## Conclusions
There are canonical 2D learning representations.  For this technique to be useful, experiments need to be conducted in 3D.  State-of-the-art approaches convert the decision surface to a 3D mesh.  However, it might be worthwhile to explore having the neural network draw directly on pixels or ray casts.  

Experiments should be conducted on learning transformations or multi-model mixtures.  For this experiment, multiple input models will be learned and indicated in the input.  The input will be (x,y,z,t), where t indicates which model to reconstruct.  This could improve compression and possibly afford interpolation between transforms.

Better reuse of intermediate output memory and calculations should be explored.  From this report, it can be seen that sharing output weights reduces performance.  However, not sharing weights means the memory and results from previous outputs go unused.  The table below shows some results from a different architecture.  Output weights are concatenated with their previous dropout layer for the next hidden layer. This could enable improved reuse.

Reconstruction Error when reusing output
Image	Hidden Layers	Width	Output	Loss Weights	Final MSE @ Output Layer
					1	2	3	4
tanstaafl	4	256	Chained	(.25,1,4,16)	0.0744	0.0172	0.0093	0.0059
Leto II				(.25,1,4,16)	0.0199	0.0038	0.0014	0.00091608

This research demonstrates that the idea of a neural network with hierarchical LOD is possible.  The final reconstructions are comparable to those produced by a straight forward neural network that cannot produce intermediate representations.  To be truly useful, experiments with this technique should be performed in 3D.  If this works well, the idea introduced in this paper may be useful for both real-time graphics and learning applications.

## Using the scripts
Directly execute the numbered scripts, for example:  python 4_shared_weights.py


