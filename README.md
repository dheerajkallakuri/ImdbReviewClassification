# IMDb Review Classification

In this project, we fine-tune BERT to perform sentiment analysis on a dataset of IMDb movie reviews. The primary goal is to train a model to classify movie reviews as either positive or negative. This project will guide you through the process of text preprocessing, model training, and evaluation using BERT, a state-of-the-art transformer-based model for natural language processing.

## About BERT

BERT (Bidirectional Encoder Representations from Transformers) and other Transformer encoder architectures have achieved remarkable success in various NLP tasks. BERT computes vector-space representations of natural language, capturing the context of each token in relation to all other tokens in the input text. The BERT model is pre-trained on a large corpus and can be fine-tuned for specific tasks.

### Model Architecture

For this project, we use a variant of BERT called Small BERT. It has the same general architecture as the original BERT but with fewer and/or smaller Transformer blocks. The architecture is as follows:

1. **Input Layer**: Accepts tokenized text input.
2. **Pre-processing Layer**: Prepares the text data for the model.
3. **Encoder Layer**: Processes the text through BERT's Transformer blocks.
4. **Dropout Layer**: Applies dropout for regularization.
5. **Dense Layer**: Final classification layer with sigmoid activation.

<img width="323" alt="Screenshot 2024-09-13 at 2 39 57 PM" src="https://github.com/user-attachments/assets/96c12f0b-e98c-4d35-ab09-7734257aa6e9">

## Dataset

We use the Large Movie Review Dataset, which contains 50,000 movie reviews from the Internet Movie Database (IMDb). This dataset is used for training and evaluating the sentiment classification model.

## Output
Here is the Image of Training and Validation loss, and Training and Validation accuracy.
<img width="1047" alt="Screenshot 2024-09-13 at 2 40 10 PM" src="https://github.com/user-attachments/assets/2182cb4d-4c89-46ff-9d95-99328547e828">


## Setup

To work on this project, you can use Google Colab. This allows for easy setup and provides access to GPU resources. The Colab notebook for this project is included in the repository.

### Colab Notebook

- [Reference Colab Notebook](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/text_models/solutions/classify_text_with_bert.ipynb)

## Learning Objectives

By the end of this project, you will:

1. Learn how to load a pre-trained BERT model from TensorFlow Hub.
2. Learn how to build a custom classification model by combining BERT with a classifier.
3. Learn how to fine-tune a BERT model on the IMDb dataset.
4. Learn how to save and use the trained model.
5. Learn how to evaluate the performance of a text classification model.
