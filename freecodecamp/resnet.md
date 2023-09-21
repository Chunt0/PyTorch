## Residual Network (ResNet) - Detailed Explanation

ResNet, or "Residual Network," is a deep neural network architecture designed to overcome challenges associated with training very deep networks. It was introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in their paper "Deep Residual Learning for Image Recognition" (2016).

### Key Concepts

- **Vanishing Gradient Problem**: Deep networks suffer from vanishing gradients during training, leading to slow learning and degraded performance.

- **Residual Learning**: ResNet introduces residual blocks with "skip connections." These blocks learn the difference between desired output and input, then add the residual to the input to get the output.

- **Identity Mapping**: Residual blocks allow identity mapping. If the desired transformation is close to identity, the residual is small, enabling easy data flow.

### How ResNet Works

1. **Vanilla Architecture and Skip Connections**: Traditional networks transform input directly. Residual blocks contain stacked layers. Output = Input + Residual.

2. **Identity Mapping**: If desired transformation is close to identity, residual is close to zero. Weights set small for easy data flow through shortcut.

3. **Advantages of Skip Connections**:
   - **Gradient Flow**: ResNet addresses vanishing gradient by allowing easy gradient flow.
   - **Deeper Networks**: ResNet can be very deep without hurting performance.
   - **Improved Accuracy**: Identity mapping helps fine-tune output, leading to better accuracy.

4. **Architectural Variants**: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152 are variants with varying depths.

5. **Prevalence in Computer Vision**: ResNet excels in image recognition tasks, achieving state-of-the-art results. Pre-trained models used for transfer learning.

6. **Extensions**: Residual learning extended to object detection, segmentation, and other domains.

### Summary

ResNet's innovation lies in skip connections and residual learning, enabling training of very deep networks. It overcomes vanishing gradient, leading to improved accuracy and performance in image recognition and beyond.

