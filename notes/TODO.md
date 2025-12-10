	##1-Compatible vision model
	- INCEPTIONV1
	- or ResNet18
	##2-salUN Integration for Unlearning
	- start with pre-trained model
	- fine-tune it on a dataset like CIFAR-10 or a subset like ImageNet
	- apply salUN to remove a target class
	- save model_before and model_after
	##3-Visualize pre- and post- unlearning with Lucent
	-use Lucent to visualize:
		- specific filters
		- class activation visuals
	- compare the resulting visuals to demmonstrate the erasure of feature representations.

	 
