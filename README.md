# adversarial-GAN

Can using GANs to do semi-supervised learning lead to models that can identify adversarial examples? 

Using the default hyperparameters and training with 100 labelled examples per class, the discriminator network has a `98%` accuracy on the MNIST test set after 300 epochs, and assigns about `85%` probability to the MNIST examples. The network has only `7.3%` adversarial examples with the [Fast Gradient Sign Method](https://arxiv.org/pdf/1412.6572.pdf) with `epsilon = 0.25`. However, it assigns only about `26%` probability to these examples being real. 

Replicate by running:
```
python mnist_GAN.py
```
