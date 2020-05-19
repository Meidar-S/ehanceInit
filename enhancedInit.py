import torch
import torch.nn.init as torchInit

import numpy as np

import math

from scipy.stats import gengamma
from scipy.stats import uniform

class UnsupportedInitMethod(ValueError):
	pass

class UnsupportedDistribution(ValueError):
	pass

######### Normal Distribution ###############

def getNormalXavierStd(tensor, gain=1.):
	fan_in, fan_out = torchInit._calculate_fan_in_and_fan_out(tensor)
	std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
	return 0, std

def getNormalKaimingStd(tensor, a = 0, mode='fan_in', nonlinearity='leaky_relu'):
	fan = torch.nn.init._calculate_correct_fan(tensor, mode)
	gain = torch.nn.init.calculate_gain(nonlinearity, a)
	std = gain / math.sqrt(fan)
	return 0, std

def enhanceNormal(tensor, dim, baseInitMethod, baseInitMethodParams):
	#Calcukate the std
	if baseInitMethod == 'kaiming':
		mean, std = getNormalKaimingStd(tensor, **baseInitMethodParams)
	elif baseInitMethod == 'xavier':
		mean, std = getNormalKaimingStd(tensor, **baseInitMethodParams)
	else:
		raise UnsupportedInitMethod('enhanceNormal.'+ str(baseInitMethod) +' unsupported method. Use \'kaiming\' or \'xavier\'')

	if dim == 0:		
		#Regular case. This means we initialize the entire tensor from same normal distribution like baseInitMethod (both )
		torchInit._no_grad_normal_(tensor, mean = 0, std = std)
		return

	for filt in tensor.data:
		#If dim is 1, then we initialize each filter
		if dim == 1:
			
			r = gengamma.rvs(a  = 0.5, c = 1, loc = 0, scale = 2*((std)**2), size=1)[0]
			
			torchInit._no_grad_normal_(filt, mean = 0, std = r**0.5)

			continue

		#If we got here, dim is 2, and initialize for each sub-filter

		#Sample variance from gamma distribution
		r = gengamma.rvs(a  = 0.5, c = 1, loc = 0, scale = 2*((std)**2), size=filt.shape[0])
		
		for subfiltIdx, subfilt in enumerate(filt):			
			torchInit._no_grad_normal_(subfilt, mean = 0, std = r[subfiltIdx]**0.5)	

	return


######### Uniform Distribution ###############

def getUniformXavierBound(tensor, gain=1.):
	fan_in, fan_out = torchInit._calculate_fan_in_and_fan_out(tensor)
	std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
	bound = math.sqrt(3.0) * std
	return bound

def getUniformKaimingBound(tensor, a = 0, mode='fan_in', nonlinearity='leaky_relu'):
	fan = torchInit._calculate_correct_fan(tensor, mode)
	gain = torchInit.calculate_gain(nonlinearity, a)
	std = gain / math.sqrt(fan)
	bound = math.sqrt(3.0) * std
	return bound

def enhanceUniform(tensor, dim, baseInitMethod, baseInitMethodParams):
	#Calcukate the bound
	if baseInitMethod == 'kaiming':
		bound = getUniformKaimingBound(tensor, **baseInitMethodParams)
	elif baseInitMethod == 'xavier':
		bound = getUniformXavierBound(tensor, **baseInitMethodParams)
	else:
		raise UnsupportedInitMethod('enhanceUniform.'+ str(baseInitMethod) +' unsupported method. Use \'kaiming\' or \'xavier\'')

	if dim == 0:		
		#Regular case. This means we initialize the entire tensor from same normal distribution like baseInitMethod (both )
		torchInit._no_grad_uniform_(tensor, -bound, bound)
		return

	for filt in tensor.data:
		#If dim is 1, then we initialize each filter
		if dim == 1:

			#According to formula of inversing uniform distribution.
			unif_1 = uniform.rvs(loc=0,scale = 1, size=1)[0]
			enhanceBound = (bound*unif_1)**2
			enhanceBound = (enhanceBound*3)**0.5
			
			torchInit._no_grad_uniform_(filt, -enhanceBound, enhanceBound)
			
			#Continue to next filter
			continue

		#If we got here, dim is 2, and initialize for each sub-filter
		#Calculate the bounds. According to formula of inversing uniform distribution.
		unif_1 = uniform.rvs(loc=0,scale = 1, size=filt.shape[0])
		enhanceBound = (bound*unif_1)**2
		enhanceBound = (enhanceBound*3)**0.5
		
		for subfiltIdx, subfilt in enumerate(filt):
			torchInit._no_grad_uniform_(subfilt, -enhanceBound[subfiltIdx], enhanceBound[subfiltIdx])
	
	return

class enhancedInit:
	def __init__(self, dim = 1, distribution = 'uniform', baseInitMethod = 'kaiming', baseInitMethodParams = dict()):
		if distribution not in ['uniform', 'normal']:
			raise UnsupportedDistribution('enhancedInit.' + str(baseInitMethod) + ' nnsupported ditribution. \'Use normal\' or \'uniform\'')

		self._weightInitFunc = enhanceUniform if distribution == 'uniform' else enhanceNormal
		self._dim = dim
		self._baseInitMethod = baseInitMethod
		self._baseInitMethodParams = baseInitMethodParams

	def initialize(self, m):
		if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d):
			self._weightInitFunc(m.weight, dim = self._dim, baseInitMethod = self._baseInitMethod, baseInitMethodParams = self._baseInitMethodParams)

		if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
			#In the linear and conv1d case dim must be 1 at max.
			self._weightInitFunc(m.weight, dim = max(self._dim, 1), baseInitMethod = self._baseInitMethod, baseInitMethodParams = self._baseInitMethodParams)	





