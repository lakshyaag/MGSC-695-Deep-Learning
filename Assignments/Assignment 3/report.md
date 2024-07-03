# Assignment 3 - Project: Multi-class Text Classification using Transformers

Submitted by: Lakshya Prakash Agarwal (261149449)

## Introduction

In this report, we explore the task of customizing and fine-tuning transformer models (specifically GPT-2) for multi-class text classification using the 20 Newsgroups dataset, which contains approximately 20,000 newsgroup documents partitioned across 20 different newsgroups. The goal is add a custom self-attention layer and a classifier head to the pre-trained GPT-2 model and fine-tune it for the specific task of multi-class text classification.

## Load and preprocess the data

### Loading the Dataset

The dataset used for this project is the 20 Newsgroups dataset, which is fetched using the `fetch_20newsgroups` function from the `sklearn.datasets` module.

### Tokenization and Preprocessing

Since we are customizing a pre-trained transformer model (GPT-2), we will use the same tokenizer that was used to pre-train the model. In this case, we use the `GPT2Tokenizer` from the HuggingFace Transformers library to tokenize the text data and pad the sequences on the left side to a maximum length.

```python
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"

def get_batch_texts(self, idx):
    # Get a batch of inputs
    return tokenizer(
        self.texts[idx],
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
```

### Data Loaders

Data loaders are created to facilitate batch processing during model training. The data is split into training, validation, and test sets, and each set is converted into a `TextDataset` object to handle the data and labels. Using a batch size of 16, the sizes are:

```python
len(train_loader), len(val_loader), len(test_loader)
> (566, 142, 471)
```

## Implement a Transformer Model

To add the necessary layers to the pre-trained GPT-2 model, we will first define the attention head, scale it to multi-head attention, add a feed-forward projection layer, and finally bundle them into a transformer block. We will then add a classifier head on top of the transformer block to perform multi-class classification.

### Building the self-attention mechanism

The attention layer moves information between positions. This is done for every token in parallel using the same parameters for each head of attention. By stacking the output of mutliple heads, we scale up to multi-head attention.

Each attention head is itself made up of two circuits:

- QK circuit: This circuit (composed of the query and key vectors) determines where to move information to and from
- OV circuit: This circuit (composed of the output and value vectors) determines what information to move.

A combination of the two circuits above allow the model to move information between tokens in the sequence, thereby "attending" to different parts of the input sequence. In terms of equations, the attention mechanism can be represented as:

$$
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Projection layer

The feed-forward layer is a simple linear layer that projects the output of the multi-head attention layer to a higher-dimensional space, applies a non-linear activation function (GELU in this case), and then projects it back to the original dimension. By doing this, the model does not move information between tokens, but rather computes using the information it has.

### The transformer block

The transformer block combines the multi-head attention and feed-forward layers. It applies layer normalization and residual connections around each sub-layer to stabilize training and improve performance.

```python
class AttentionHead(nn.Module):
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        weights = q @ k.transpose(-2, -1) * (C**-0.5)

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        return out

class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.sa_heads = MultiHeadAttention(n_heads, n_embed // n_heads)
        self.ln1 = nn.LayerNorm(n_embed)

        self.ffwd = FeedForward(n_embed, 4)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))  # Residual connection + attention
        x = x + self.ffwd(self.ln2(x))  # Residual connection + feed-forward
        return x
```

### Adding the classifier head

The classifier head is added as the final layer of the model. It takes the output of the transformer block and applies a linear transformation followed by a softmax activation to obtain the class probabilities.

## Train the model

The model is trained using the PyTorch Lightning framework, which simplifies the training loop and other boilerplate code. The training involves feeding batches of character sequences to the model and using cross-entropy loss to compare the model's predictions with the actual labels.

The hyperparameters used for training are:

```python
BATCH_SIZE = 16
MAX_LENGTH = 512

LEARNING_RATE = 1e-5

N_EMBED = 768
N_HEADS = 2
N_BLOCKS = 12
DROPOUT = 0.2
NUM_LABELS = 20
OPTIMIZER = "AdamW"
```

## Evaluate the model

After training the model, we evaluate its performance on a held-out test set. We calculate the classification accuracy as the primary metric. The accuracy on the test set is 78%.

### Example classification

#### Input

```text
From: Rick Miller <rick@ee.uwm.edu>
Subject: X-Face?
Organization: Just me.
Lines: 17
Distribution: world
NNTP-Posting-Host: 129.89.2.33
Summary: Go ahead... swamp me.  <EEP!>

I'm not familiar at all with the format of these "X-Face:" thingies, but
after seeing them in some folks' headers, I've *got* to *see* them (and
maybe make one of my own)!

I've got "dpg-view" on my Linux box (which displays "uncompressed X-Faces")
and I've managed to compile compface too... but now that I'm *looking*
for them, I can't seem to find any X-Face:'s in anyones news headers!  :-(

Could you, would you, please send me your "X-Face:" header?

I *know* I'll probably get a little swamped, but I can handle it.

        ...I hope.

Rick Miller  <rick@ee.uwm.edu> | <ricxjo@discus.mil.wi.us>   Ricxjo Muelisto
Send a postcard, get one back! | Enposxtigu bildkarton kaj vi ricevos alion!
          RICK MILLER // 16203 WOODS // MUSKEGO, WIS. 53150 // USA

```

#### Output

```text
Actual label: comp.windows.x
Predicted label: comp.windows.x
```

## Conclusion

This report detailed the process of customizing and fine-tuning GPT-2 for text classification with a custom self-attention layer added to the model. Future work could explore further scaling up the model, experimenting with different hyperparameters, or using other transformer architectures like BERT.
