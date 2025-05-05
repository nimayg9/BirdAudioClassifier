# BirdAudioClassifier

Introduction
Without being a biologist, identifying the exact species of birds chirping, insects buzzing, and frogs croaking in the middle of a rainforest in Colombia is almost an impossible task. These ecosystems are extremely biodiverse, with hundreds of different species coexisting, many of which sound very similar to the human ear.

Traditionally, biologists and field researchers had to rely on manual audio analysis and expert knowledge to identify species, which can be time-consuming and not very practical. This inspired us to create a neural network to help identify birds based on the sounds they make.

We decided to feed it labeled audio recordings of different bird calls and train the model to recognize bird species native to the Middle Magdalena Valley of Colombia.

Our goal was to build a network that could accurately identify the bird species present in a short audio clip. In this post, I will describe our methodology, data, results, discussion, and findings.

Methodology and Data
We started with the BirdCLEF 2025 dataset from Kaggle, which contains a large collection of labeled sound recordings from a wide range of species. These labeled audio files were essential for training our neural network to recognize and classify different bird sounds.

The raw audio data was messy, with some files being inconsistent, so it had to be cleaned before use. We filtered out tiny classes and standardized everything to the same format and shape. We also used a tool called YAMNet from TensorFlow Hub to convert audio into numerical summaries that capture the key patterns and tones in each clip. These numerical summaries were 1024-dimensional embeddings, which served as the input to our neural network.

Our model architecture included:
An input layer that receives a 1024-dimensional vector from YAMNet

Three dense (fully connected) layers, each with 200 units and ReLU activation

Dropout layers with a rate of 0.1 placed after each dense layer to help prevent overfitting

An output layer with softmax activation to handle multi-class classification across all species

We also implemented early stopping with a patience of three epochs to halt training when validation performance stopped improving.

Results and Findings
When training and testing the model, we achieved an accuracy of around 35 percent with 206 classes on the test data. This is a strong result, considering the large number of species, how similar some sounds are, and the short length of the audio clips. We did observe some overfitting despite using dropout.

Discussion
One of the main challenges we faced was the imbalance in data across species. Some had hundreds of audio clips, while others had only a few. To address this, we manually limited the number of samples per class to get fairer results. We also applied early stopping and regularization using dropout to help the model generalize better to new bird audio clips.

Conclusion
Our project showed that training a neural network to classify birds based on short audio clips, even in biodiverse environments, is possible. Using YAMNet embeddings and a simple neural network architecture, we achieved a test accuracy of 35 percent with 206 classes, which is promising given the complexity of the task.

We faced challenges related to imbalanced data and small sample sizes for certain species, but regularization techniques such as dropout and early stopping helped improve the model's generalization. Our results suggest that with more data and stronger models, this approach could become even more effective and serve as a valuable tool for biodiversity research. There are many future improvements that could be made using more advanced architectures.

Data is from: https://www.kaggle.com/competitions/birdclef-2025
