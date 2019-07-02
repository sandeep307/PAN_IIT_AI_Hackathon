## Approach
1. Given the very small number of images (5k only) , I split the data set into 80% - 20%
ratio. The first part with 80% images (~4000 images) along with augmented images was
used for training and rest 20% (~1000 images) was used for validation.
2. I used image augmentations like adding hue saturation, additive gaussian noise ,
blurring, sharpen, grayscale etc to increase number of images so that each class will
have at least 8k training images. Models trained on this augmented dataset gave
significant boost to accuracy (by ~2%) .
3. I tried different CNN architectures with (pretrained ImageNet weights) like ResNet-18,
ResNet-34, ResNet-50, ResNet-101, Inception-V3, DensNet-121 in PyTorch.
4. For training I used SGD with momentum with decay learning rate.
5. Inception-V3 and Resnet-50 performed best amongst all the models so the final
submission was ensemble of these 2 models. To create ensemble I just averaged
probability values obtained from all the models and then took the category with highest
probability.
