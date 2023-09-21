# Quantization_adaptiveround
Implementing Quantization on resnet18, using a mini dataset and reconstruct the model.
https://mqbench.readthedocs.io/en/latest/user_guide/PTQ/advanced.html

Implementing post training quantization technique using symmetric fake quantization on resnet18.
Reconstruction of the model can be done in lower bit upto 4 bit for reasonable results.
Quantizing weights and activations per layer or per block.
Weights and activations are implemented using different algorithms
1. Adaptive rounding based on the output of each layer/block
2. Block/Layer reconstruction.

