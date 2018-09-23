# Hydrangia
Code for the Kaggle Invasive Species Monitoring Competition.

Notebooks train Xception and DeepNet201 convolutional architectures.

DeepNet is trained on the top layer first, and then all layers. Xception is trained on all layers.

Train/Validation spilt is 80/20.

Validation is measured using AUC.

Results are very similar:

Xception - 0.98578 AUC score

DeepNet201 - 0.98827 AUC score
