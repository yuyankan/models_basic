# 1. u-net link: https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/
# 2. spacial unet link: https://www.researchgate.net/figure/U-Net-with-late-spatio-temporal-fusion-On-the-bottom-right-frame-the-Trainable-Temporal_fig1_340989612

# U-Net Architecture Explained
Last Updated : 23 Jul, 2025

U-Net is a kind of neural network mainly used for image segmentation which means dividing an image into different parts to identify specific objects for example separating a tumor from healthy tissue in a medical scan. The name “U-Net” comes from the shape of its architecture which looks like the letter “U” when drawn. It is widely used in medical imaging because it performs well even with a small amount of labeled data.

## U-Net Architecture
The architecture is symmetric and has three key parts:

### Contracting Path (Encoder):

- Uses small filters (3×3 pixels) to scan the image and find features.
- Apply an activation function called ReLU to add non-linearity help the model to learn better.
- Uses max pooling (2×2 filters) to shrink the image size while keeping important information. This helps the network focus on bigger features.

### Bottleneck:

The middle of the “U” where the most compressed and abstract information is stored. It links the encoder and decoder.

### Expansive Path (Decoder):

- Uses upsampling i.e increasing image size to get back the original image size.
- Combines information from the encoder using “skip connections.” These connections help the decoder get spatial details that might have been lost when shrinking the image.
- Uses convolution layers again to clean up and refine the output.
  ![unet architecture](u-net.jpg)
