# CrowdLayer
A neural network layer that enables training of deep neural networks directly from crowdsourced labels (e.g. from Amazon Mechanical Turk) or, more generally, labels from multiple annotators with different biases and levels of expertise, as proposed in the paper:

> Rodrigues, F. and Pereira, F. Deep Learning from Crowds. In Proc. of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18).

This implementation is based on Keras and Tensorflow.

# Usage

## Classification

Using the crowd layer in your own Keras deep neural networks for classification problems is very simple. For example, given a sequential model in Keras, you just need to add a "CrowdsClassification" layer as the last layer of the model (on top of what would normally be your output layer, e.g. "Dense" with softmax activation) and use a specialized cross-entropy loss to handle missing answers (encoded with "-1"): 

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

Once the network is trained, you can remove the crowd layer from the model, exposing the bottleneck layer, and using it to make predictions:

```python
# remove crowd layer before making predictions
model.pop() 
model.compile(optimizer='adam', loss='categorical_crossentropy')

# make predictions
predictions = model.predict(...)
```

For details, kindly see the [paper](http://www.fprodrigues.com/publications/deep-crowds/). 

## Regression

Using the crowd layer in your own Keras deep neural networks for regression problems is very similar to the classification case: 

```python
# build your own base model for regression
model = Sequential()
model.add(...) 
...

# add crowd layer on top of the base model
model.add(CrowdsRegression(N_ANNOT, conn_type="B"))

# instantiate specialized masked loss to handle missing answers
loss = MaskedMultiMSE().loss

# compile model with masked loss and train
model.compile(optimizer='adam', loss=loss)

# train the model
model.fit(...)
```

Once the network is trained, you can remove the crowd layer from the model, exposing the bottleneck layer, and using it to make predictions.

## Sequence labelling

Using the crowd layer in your own Keras deep neural networks for sequence labelling problems is very similar to the classification case, but since the output are now sequences, you need to use the following loss function instead:

```python
# instantiate specialized masked loss to handle missing answers
loss = MaskedMultiSequenceCrossEntropy(N_CLASSES).loss
```

# Demos

For demonstration purposes, we provide 4 practical applications of the crowd layer in the following problems:

* Image classification (binary) using simulated annotators on the [Cats vs Dogs dataset](https://www.kaggle.com/c/dogs-vs-cats);
* Image classification (multi-class) using real annotators from Amazon Mechanical Turk on [LabelMe data](http://labelme.csail.mit.edu/Release3.0/browserTools/php/dataset.php);
* Text regression using real annotators from Amazon Mechanical Turk on the [MovieReviews dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/);
* Sequence labelling using real annotators from Amazon Mechanical Turk on the [2003 CONLL Named Entity Recognition (NER) dataset](https://cogcomp.org/page/resource_view/81).

See the corresponding jupyter notebooks available on the repository. 

# Datasets

The datasets used in all the experiments from the demos are available [here](http://www.fprodrigues.com/publications/deep-crowds/).



