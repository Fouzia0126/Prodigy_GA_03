# 🎨 Neural Style Transfer with PyTorch

Apply the artistic style of one image (e.g., a famous painting) to the content of another image using neural style transfer, implemented in PyTorch.

## 🧠 How It Works

This project uses a pre-trained [VGG19](https://arxiv.org/abs/1409.1556) convolutional neural network to extract features from two images:
- A **content image**, whose structure you want to preserve.
- A **style image**, whose artistic style you want to apply.

The network then optimizes a target image to match the content features of the content image and the style features (via Gram matrices) of the style image.

---

## 📁 Project Structure

