# CrowdLayer
A neural network layer that enables training of deep neural networks directly from crowdsourced labels (e.g. from Amazon Mechanical Turk) or, more generally, labels from multiple annotators with different biases and levels of expertise, as proposed in the paper:

> Rodrigues, F. and Pereira, F. Deep Learning from Crowds. In proc. of the thirty-second AAAI Conference on Artificial Intelligence (AAAI-18).

This implementation is based on Keras and Tensorflow.

# Usage

## Classification

Using the crowd layer in your own Keras deep neural networks for classification problems is very simple. For example, you have a sequential model, you just need to add the 

```python
# build your own base model for classification
model = Sequential()
model.add(...) 
...

# add crowds layer on top of the base model
model.add(CrowdsClassification(N_CLASSES, N_ANNOT, conn_type="MW"))

# instantiate specialized masked loss to handle missing answers
loss = MaskedMultiCrossEntropy().loss

# compile model with masked loss and train
model.compile(optimizer='adam', loss=loss)

# train the model
model.fit(...)
```

# Demos



