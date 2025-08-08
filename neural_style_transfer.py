import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    image = in_transform(image).unsqueeze(0)
    return image.to(device)

def imshow(tensor, output_path=None):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    if output_path:
        image.save(output_path)
    else:
        plt.imshow(image)
        plt.axis('off')
        plt.show()

def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def main(content_path, style_path, output_path="output.jpg", epochs=300):
    content = load_image(content_path)
    style = load_image(style_path, shape=content.shape[-2:])

    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)

    style_weights = {
        'conv1_1': 1.0,
        'conv2_1': 0.75,
        'conv3_1': 0.5,
        'conv4_1': 0.25,
        'conv5_1': 0.1
    }

    content_weight = 1e4
    style_weight = 1e2
    optimizer = optim.Adam([target], lr=0.003)

    for i in range(1, epochs + 1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            _, d, h, w = target_feature.shape
            style_loss += layer_style_loss / (d * h * w)
        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f"Epoch {i}, Loss: {total_loss.item():.4f}")

    imshow(target, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python neural_style_transfer.py <content_image_path> <style_image_path> [output_image_path]")
    else:
        content_path = sys.argv[1]
        style_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "output.jpg"
        main(content_path, style_path, output_path)
