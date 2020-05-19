# Enhanced Initialize

Weight initialization, although sometimes overlooked, is a very important part in the training process. *enhanceInit* implements a suggestion for a new initialization method, which recieves other distribution as input and 'enhance' it.

## Quick Use

You can use this enhancment fast and easily. In the common case that you're using a network with ReLU, just add these lines to you code:
```Python
from enhancedInit import enhancedInit
net.apply(enhancedInit().initialize)
```
Try this, and there's a good change you'll get better result or faster convergence without effort!

## Short Background

Weight initialization in neural networks is important in order to keep the variance from decaying in deeper layers. Currently the must popular initialization methods are *kaiming* and *xavier*, yet both are values drawn from uniform/normal distribution, where each method calculated different parameters for the distribution.
The enhancment suggested in this work uses [compound distribution](https://en.wikipedia.org/wiki/Compound_probability_distribution), so that for a given initialization distribution provided as input, it initialize sub-tensors of the weight tensor of each layer (Conv/Linear) with different distribution. Although it means that now the initialization distribution for each sub-tensor is not as the original, the original weight tensor (which contains all of the sub-tensors) still do. Yet, now the original weight tensor has more "variaty" in it (e.i. moments higher than 2nd has changed). 
As shown, for very deep neural networks, this improves not only the final result, but also provides much faster convergence. This method can also be implemented for every symmetrical and centered distribution, althought currently it only normal and uniform are implemented (you'll need to manually calculate the pdf inverse, as shown).

## Usage and Examples

```Python
import torchvision
from enhancedInit import enhancedInit

#Create some net for example
net = torchvision.models.resnet18()

"""
Create the enhancedInit object. 
	dim : Can be 0, 1, 2. 
		0 - Initialize on the entire layer, so no enhancment is done.
		1 - Initialize each tensor in dimension 1. e.g. in Conv1d it means each filter, and in Linear it means each row
		2 - initialize each tensor in dimension 2. e.g. in Conv2d it means each "row" in the 3-dimensional filter.

	distribution : 'uniform' or 'normal'. See section 3.2 in the pdf to understand how to implement for costume symmetric and centered distributions.

	baseInitMethod : 'kaiming' or 'xavier'. Which method to enhance

	baseInitMethodParams : Parameters for the initialization method. If empty, Used torch defaults.
"""
enhance_kaiming_uniform_dim1 = enhancedInit(dim = 1, distribution = 'uniform', baseInitMethod = 'kaiming')

#Apply the initialization on each module of the network
net.apply(enhance_kaiming_uniform_dim1.initialize)

#Continue training
```

