import torch
import torchvision
from enhancedInit import enhancedInit

import time
def checkNan(m):
	if hasattr(m, 'weight'):
		if torch.isnan(m.weight).any():
			print("Bad initialization with NaN")

def main():
	#Network to test upon
	net = torchvision.models.resnet18()
	
	for dim in [0,1,2]:
		for distribution in ['uniform', 'normal']:
			for baseInitMethod in ['kaiming', 'xavier']:

				#Create the enhanceInit object
				ei = enhancedInit(dim = dim, distribution = distribution, baseInitMethod = baseInitMethod)
				
				#Apply on network
				s_time = time.time()
				net.apply(ei.initialize)
				e_time = time.time()

				#Check for nans
				net.apply(checkNan)

				print("Done - ", dim, distribution, baseInitMethod, ".Took:", e_time - s_time)

	return	

if __name__ == '__main__':
	main()