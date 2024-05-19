# Assignment 2: Text Generation with RNNs (Duration: 1 week)

## Objective

This assignment aims to deepen your understanding of Recurrent Neural Networks (RNNs), particularly their application in sequence generation tasks. You'll implement an RNN and use it to generate text data in a specific style.

## Dataset

A dataset of Shakespeare's writings. This is a widely used dataset for text generation and can provide interesting results.

## Tasks

- Preprocess the text data: The text data needs to be tokenized, possibly with additional steps like lowercasing and punctuation removal. You'll also need to convert the text data into sequences that your RNN can learn from.
- Implement an RNN: Using your chosen deep learning framework, implement an RNN, LSTM, or GRU for this task. Decide on aspects such as the number of layers, hidden units, etc.
- Train your model: Train the model using your processed data. Make sure to implement a mechanism to save the weights of the model periodically or when it achieves the best performance on a validation set.
- Generate new text: Using your trained model, generate new text that mimics the style of the training corpus.

## Success Criteria

The success of this assignment will be determined by:

- Proper data preprocessing to transform raw text data into a format suitable for RNNs.
- Successful implementation and training of an RNN.
- The generated text should reasonably mimic the style of the training corpus. While we don't expect perfect results, the generated text should show some coherent structure and stylistic similarities with the original text.
- Clear documentation of your architecture decisions, training process, and generation method.
