# What is Embedding?

It's the first layer in the architecture of Encoder-Decoder Transformer.

But before we jump into embedding, let’s quickly understand where they fit in.

Imagine you’re typing a sentence into Google Translate:\
\> "Hello world"

The model behind the app doesn’t directly process this raw text. Instead, it first **tokenizes** the sentence — converting words or subwords into unique numeric IDs known as **tokens**.

If you're unfamiliar with tokenization, check out this beginner-friendly tutorial:\
[Tokenization Explained](https://www.youtube.com/watch?v=zduSFxRajkE)

## Embedding: From Token IDs to Vectors

Once we have token IDs, we pass them through an **embedding layer** — a kind of lookup table that maps each token ID to a dense vector of real numbers.

This gives us: Words -\> Token IDs -\> Embedding Vector.

The output of the embedding layer has a shape of: `sequence length` \* `embedding dimension`

`sequence length`: number of tokens

`embedding dimension`: depends on model (e.g. in GPT-2 it's 768)

A tabular view of the token to embedding process:

| Word  | Token ID | Embedding Vector (dim=4 example) |
|-------|----------|----------------------------------|
| Hello | 101      | `[0.25, -0.11, 0.78, 0.03]`      |
| World | 432      | `[0.90, 0.12, -0.55, 0.67]`      |

(Values shown here are simplified examples — not actual outputs.)

# Positional Encoding Layer

The **Positional Encoding** layer helps the model understand the **order of tokens** in a sequence. Since Transformer models process all tokens in parallel (unlike RNNs), they don't inherently know the position of each token. This layer injects **positional information** into the model.

## Why is this important?

Without positional encoding, the model would treat the following two sentences as the same bag of words:

-   "The cat sat on the mat"

-   "The mat sat on the cat"

Clearly, the meaning is different — but without knowing the order of words, a model can't tell.

## How does it work?

The process is: To generate the positional vector, we use **sine and cosine functions** at different frequencies:

PE(pos, 2i) = sin(pos / 10000\^(2i / d_model))

PE(pos, 2i+1) = cos(pos / 10000\^(2i / d_model))

Where:

`pos` = token position in the sequence (e.g., 0 to max sequence length)

`i` = the dimension index (0 to embedding dimension - 1)

`d_model` = the model’s embedding size (e.g., 512 or 768)

## What does this produce?

Each **position** is assigned a unique vector of size `d_model`. These vectors are then **added to the token embeddings**, allowing the model to know both **what** a token is and **where** it is.

Here's an example table for illustration (assuming `embedding_dim = 4`):

| Position | PE\[0\] (sin) | PE\[1\] (cos) | PE\[2\] (sin) | PE\[3\] (cos) |
|----------|---------------|---------------|---------------|---------------|
| 0        | 0.000         | 1.000         | 0.000         | 1.000         |
| 1        | 0.841         | 0.540         | 0.010         | 1.000         |
| 2        | 0.909         | -0.416        | 0.020         | 1.000         |

(Values shown here are simplified examples — not actual outputs.)

# Where's code?

Check out the pytorch-code.py file under the same directory

# What's next:

All about attention!