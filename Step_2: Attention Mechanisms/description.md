# Attention Mechanisms

Attention mechanism gives the ability to the model to **focus on important tokens and their relationships** which improves text generation.

## Self Attention

This mechanism enables the model to:

-   Weight token importance; when looking at each token in the given input, it assigns higher weight to the most relevant tokens

-   Capture long-range dependencies; if you pass a thousand word long input, the model can identify the relationship between tokens, regardless of their distance - e.g. if a word on the top is highly relevant to another word at the bottom, it can identify that relationship.

For example in the "The cat sat on the mat" sentence, this mechanism would assign the weight like below:

| Word/Token | Weight |
|------------|--------|
| The        | 0.05   |
| cat        | 0.60   |
| sat        | 0.25   |
| on         | 0.05   |
| the        | 0.03   |
| mat        | 0.02   |

Let's say, the model tries to find the relationship of "cat" against all other words, in this case it'd identify the "sat" as the most relevant word to "cat".

## Multi-Head (Self) Attention

Think of this layer as multiple smart people in a team that each member takes a distinct responsibility in tackling a problem. Each head would act like a member of the team. This will lead to richer representation of the whole input and enhance LLM's effectiveness across the tasks.

Example:

In a sentence like: "The bank will not lend money to the farmer near the river."

-   **Head 1** might focus on financial sense of "bank” → attends strongly to "lend” and "money.”

-   **Head 2** might focus on physical location → attends from "bank” to "river.”

-   **Head 3** might connect the actor "bank” to the recipient "farmer.”

-   **Head 4** might track the negative sentiment → attends from "not” to "lend.”

When combined, the model knows:

-   "bank” here has *two possible meanings* but the financial one is stronger in context.

-   There’s still a subtle spatial connection to "river,” which could matter in some tasks.

**Do we need to assign the tasks manually to each head?** No, the model takes care of assigning tasks to heads on its own.

**The Transformer uses Query, Key and Value** to achieve self-attention.

-   Query: Indicates what each token "is looking for" in other token?

-   Key: Represents content of each token

-   Value: Is the actual content of each token

Let's see it in a real world example:

You walk into a huge library, but instead of searching through all the shelves yourself, you ask the librarian for help.

The roles: Query (Q) → Your question

-   This represents what you’re looking for.

-   In Transformers, each token creates its own “query” to figure out what it needs from other tokens.

Key (K) → Labels on the library catalog

-   Each book in the library has a “key” — metadata about what it contains.

-   In Transformers, each token has its own “key” that describes what information it holds.

Value (V) → The actual book content

-   Once the librarian finds which books are relevant (by matching your query to keys), they pull out the actual content — the values — to give you the answer.

-   In Transformers, “values” hold the actual representation or meaning that will be passed on to the next layer.

In the next section, we'll explore the mechanics of how the model chooses the most relevant tokens. In above example, the right book.

## Attention Scores

Alright, fun stuff; to identify the relevant tokens, we need to do a little math - that's calculating the similarity between Q and K. How do we do this?

-   Think of **Query (Q)** and **Key (K)** as two vectors (lists of numbers) that represent meaning in a high-dimensional space.

-   To see *how similar* they are, we **take their dot product**: multiply each number in Q by the corresponding number in K, then add them all up.

-   Big result → more similar → higher attention score.

-   Then we scale it (so numbers don’t get too big) and run it through a **softmax** to turn all scores into probabilities that sum to 1. This step creates the **attention weight** which we'll explore in the next section in more detail.

mathy view:

If Q = [q1,q2,q3] and K = [k1,k2,k3]:

similarity = q1​⋅k1​+q2​⋅k2​+q3​⋅k3

dot product: We use the dot product between Q and K to quickly and efficiently measure their alignment in meaning-space. High dot product means high relevance, which translates to more attention on that token.

softmax: An activation function that adds non-linearity to the model to cover complex relationships

## Attention Weights

Attention weights reflect the relevance or attention that model assigns to each token. For example in "Harry Potter is my favorite book" in a library context, the attention weight would look like below:

|                  |              |      |      |          |      |
|------------------|--------------|------|------|----------|------|
|                  | Harry Potter | Is   | My   | Favorite | Book |
| Query            | Harry Potter |      |      |          |      |
| Attention Weight | 0.21         | 0.03 | 0.05 | 0.31     | 0.40 |

When looking for "Harry Potter", the words "Favorite" and "Book" have the highest relevancy or attention.

In the context of library, the model predicts the "Harry Potter" as a book rather than a movie.

# Where's code?

Check out the pytorch-code.py file under the same directory

# What's next:

Encoder Architecture!