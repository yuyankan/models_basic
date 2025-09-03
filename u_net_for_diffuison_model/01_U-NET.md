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


The above image shows U-Net turning a 572×572 image into a smaller 388×388 segmented map. It shrinks the image to capture features then upsamples to restore size using skip connections to keep details. The output labels each pixel as object or background.

## How U-Net Works
After understanding the architecture, it’s important to see how U-Net actually processes data to perform segmentation:

- **Input Image**: The process starts by feeding a medical or other input image typically grayscale into the network.
- **Feature Extraction (Encoder)**: The encoder extracts increasingly abstract features by applying convolutions and downsampling. At each level the spatial size decreases while the number of feature channels increases and allow the model to capture higher-level patterns.
- **Bottleneck Processing**: This is the middle part of the network where the image is reduced the most. It holds a small but very meaningful version of the image that captures the main features.
- **Reconstruction and Localization (Decoder)**: The decoder begins to reconstruct the original image size through upsampling. At each level it combines decoder features with corresponding encoder features using skip connections to retain fine-grained spatial details.
- **Skip Connections for Precision**: Skip connections help preserve spatial accuracy by bringing forward detailed features from earlier layers. These are especially useful when the model needs to distinguish boundaries in segmentation tasks.
- **Final Prediction**: A 1×1 convolution at the end converts the refined feature maps into the final segmentation map where each pixel is classified into a specific class like foreground or background. This output has the same spatial resolution as the input image.
