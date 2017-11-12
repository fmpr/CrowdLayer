# CrowdLayer
A neural network layer that enables training of deep neural networks directly from crowdsourced labels (e.g. from Amazon Mechanical Turk) or, more generally, labels from multiple annotators with different biases and levels of expertise, as proposed in the paper:

> Rodrigues, F. and Pereira, F. Deep Learning from Crowds. In Proc. of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18).

This implementation is based on Keras and Tensorflow.

# Usage

## Classification

Using the crowd layer in your own Keras deep neural networks for classification problems is very simple. For example, you have a sequential model, you just need to add a "CrowdsClassification" layer as the last layer of the model (on top of what would normally be your output layer, e.g. "Dense" with softmax activation) and use a specialized cross-entropy loss to handle missing answers. 

```python
# build your own base model for classification
model = Sequential()
model.add(...) 
...

# add crowd layer on top of the base model
model.add(CrowdsClassification(N_CLASSES, N_ANNOT, conn_type="MW"))

# instantiate specialized masked loss to handle missing answers
loss = MaskedMultiCrossEntropy().loss

# compile model with masked loss and train
model.compile(optimizer='adam', loss=loss)

# train the model
model.fit(...)
```

Once the network is trained, you can remove the crowd layer from the model, exposing the bottleneck layer, and using it to make predictions.

```python
# remove crowd layer before making predictions
model.pop() 
model.compile(optimizer='adam', loss='categorical_crossentropy')

# make predictions
predictions = model.predict(...)
```

For details, kindly see the [paper](http://www.fprodrigues.com/publications/deep-crowds/). 

# Demos



