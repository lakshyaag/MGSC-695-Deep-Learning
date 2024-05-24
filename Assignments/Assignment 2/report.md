# Assignment 2: Text Generation using LSTM Networks

Submitted by: Lakshya Prakash Agarwal (261149449)

## Introduction

In this report, we explore the task of text generation using a character-level LSTM network. The dataset used is a collection of Shakespeare's writings, which provides a rich and complex example of English literature. The goal is to train a model that can generate text in the style of Shakespeare.

## Load and Preprocess Data

### Loading the Dataset

The dataset is loaded from a text file containing Shakespeare's writings. The text is processed to create sequences of characters, which are then used to train the LSTM model.

```python
# Load the Shakespeare dataset
s = ShakespeareDataset("./input.txt", seq_len=SEQ_LEN)

# Preprocessing the text
self.chars = sorted(list(set(text)))
self.vocab_size = len(self.chars)
self.s_to_i = {s: i for i, s in enumerate(self.chars)}
self.i_to_s = {i: s for i, s in enumerate(self.chars)}
```

### Data Loaders

Data loaders are created for the training and validation sets to facilitate batch processing during model training.

```python
# Create data loaders

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
```

## LSTM Model Architecture

The LSTM model architecture consists of an embedding layer, several LSTM layers, and a fully connected output layer. The model is designed to predict the next character in a sequence, given the previous characters.

### Breaking it down

There's two key ideas in the LSTM / RNN architecture, namely sequential processing and the hidden state.

#### Sequential processing

This is the fundamental idea behind RNNs. The model processes the input sequence one token at a time, and maintains a hidden state that is updated at each time step. The hidden state is used to maintain information about the sequence that the model has seen so far. This is the reason we created those sequences of X and y earlier.

In the forward pass, we loop over the input sequence, and at each time step, we pass the input token, along with the hidden state from the previous time step through the various gates of the LSTM cell to get the output token and the updated hidden state.

#### Hidden state

The hidden state in a RNN-arhitecture is a vector that acts as a memory for the model. What makes LSTMs so performant is that it implements two versions of the memory, namely the short-term memory (what the model is currently looking at, $h_t$) and the long-term memory (what the model has seen so far or the cell state, $c_t$).

The gates in the LSTM cell control the flow of information in the computational graph. Through this, each cell can interpolate between the short-term and long-term memory, and decide what information to keep and what to discard. Let's look at the gates:

- Forget gate: This gate decides what information to keep and what to discard from the long-term memory. It takes the input token ($x_t$) and the hidden state from the previous time step ($h_{t-1}$) as input, and outputs a vector of values between 0 and 1 that are used to scale the long-term memory, essentially deciding what to forget (close to 0) and what to keep (close to 1).
- Input gate: This gate combines information from the input token and $h_{t-1}$ and decides what new information to add to the long-term memory. Here, the input token is passed through a sigmoid function to decide what to add to the long-term memory, and a tanh function to decide what to add.
- Output gate: This gate decides what information to take from the long-term memory and pass to the short-term memory. The output gate also takes $x_t$ and $h_{t-1}$ as input and outputs a vector to scale the long-term memory's contribution to the short-term memory.

##### The equations

Mathematically, the flow of an LSTM cell can be described as follows:

$$
\begin{align}
f_t &= \sigma(W_f \cdot [x_t, h_{t-1}] + b_f) \\
i_t &= \sigma(W_i \cdot [x_t, h_{t-1}] + b_i) \\
o_t &= \sigma(W_o \cdot [x_t, h_{t-1}] + b_o) \\
g_t &= \tanh(W_g \cdot [x_t, h_{t-1}] + b_g) \\
\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
h_t &= o_t \odot \tanh(c_t)
\end{align}
$$

The $\odot$ operator denotes element-wise or point-wise multiplication. The $W$ and $b$ are the weights and biases of the LSTM cell, respectively. The $[x_t, h_{t-1}]$ notation denotes the concatenation of the input token and the hidden state from the previous time step.

### Architecture

- **Embedding Layer**: Maps each character to a high-dimensional vector.
- **LSTM Layers**: Processes the sequence of embeddings to capture temporal dependencies.
- **Fully Connected Layer**: Outputs the logits for each character in the vocabulary.

## Training and Validation

The model is trained using the PyTorch Lightning framework, which simplifies the training loop and other boilerplate code. The training involves feeding batches of character sequences to the model and using cross-entropy loss to compare the model's predictions with the actual next characters.

### Training Process

- **Loss Function**: Cross-entropy loss, suitable for classification tasks.
- **Optimizer**: Adam optimizer with a learning rate of 1e-3 and weight decay of 1e-6.
- **Callbacks**: ModelCheckpoint to save the best model and LearningRateMonitor to log the learning rate.

## Evaluation and Text Generation

After training, the model is evaluated on the validation set, and its ability to generate text is demonstrated by sampling from the model.

### Evaluation

The model's performance is evaluated based on the loss on the validation set. The current model achieves a validation loss of 1.22, indicating that it can predict the next character with reasonable accuracy.

### Text Generation

The model generates text by sampling characters according to the probabilities predicted by the model, given an initial seed text. The sampling process involves choosing the character with the highest probability at each step, which can lead to repetitive or predictable text.

#### Example Output

```
CORIOLANUS:
Fellow, she is not one.

POLIXENES:
To you, mispredition man, more than this again; come, sit down;
For having hours discreeth,
Whose case is marvellous chance to make the cold; by the said was Bohemia; thou hast masquing blush and presence at the matter.

MENENIUS:
A hundred thousand thoughts of yonder: if things expect'd?

DUKE OF YORK:
Lay her volubtle, sir, and you hope I see the assembling slave:
But that I want you do protest.
```

## Conclusion

This report detailed the process of training a character-level LSTM network to generate text in the style of Shakespeare. The model demonstrates the capability of LSTMs to capture complex language patterns and generate coherent and stylistically similar text. Future work could explore deeper architectures, different datasets, or hybrid models that combine LSTMs with other types of neural networks.
