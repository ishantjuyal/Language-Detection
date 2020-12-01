# Language Detection

### Dataset

This particular dataset consists of common sentences in 4 different languages. The four languages are: German (Deutsch), French, Spanish, and English.

The sentences are already preprocessed so you don't need to do any preprocessing if you are going to use the dataset.

Inside each CSV file, we have two columns:

text: This column contains the sentences in different languages

language: As the name gives it away, this column contains the language to which the sentence belongs to.

You can access the dataset [here.](https://github.com/ishantjuyal/Language-Detection/tree/main/Dataset)

### Approach

I used Neural Networks (without any LSTMs) to train the modelling. I did use embeddings for converting word tokens to vectors/ embeddings.

### Performance

Accuracy of model on test set:
0.9987822204181043

You can see the demo in the notebook [here.](https://github.com/ishantjuyal/Language-Detection/blob/main/Language%20Detection%20NN%20Model.ipynb)

### Dataset Reference

[Kaggle Dataset](https://www.kaggle.com/ishantjuyal/language-detection-dataset)