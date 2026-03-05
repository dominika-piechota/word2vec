# Word2Vec from Scratch (Pure NumPy)

This repository contains my implementation of the core Word2Vec training loop using **strictly pure NumPy**. No PyTorch, TensorFlow, or any other high-level ML frameworks were used. 

The goal of this project was to implement the optimization procedure and understand the underlying math of word embeddings by building the **Skip-Gram with Negative Sampling (SGNS)** architecture from the ground up.

## Implementation Details
Based on the assignment requirements, the core of the algorithm (`model.py`) includes:
* **Algorithm:** Skip-Gram with Negative Sampling (SGNS).
* **Optimization:** Custom forward pass, binary cross-entropy loss calculation for SGNS, and hand-derived gradients (backward pass).
* **Vectorization:** The backward pass avoids slow `for` loops by utilizing NumPy's dot products and array broadcasting to calculate gradients for the context and negative samples simultaneously.
* **Parameter Updates:** Vanilla Stochastic Gradient Descent (SGD).
* **Negative Sampling:** Implemented the unigram distribution raised to the 3/4ths power (Mikolov et al., 2013) for optimal negative sample drawing using a precomputed 1M-element probability array for O(1) sampling time.

## Custom Dataset
For the training corpus, I chose a custom text dataset consisting of three books by Ernest Hemingway (raw `.txt` files). The project features a custom data pipeline (`dataset.py`) that:
1. Cleans and tokenizes raw text.
2. Builds a vocabulary and filters out rare words (min_count).
3. Yields training batches `(center_word, context_word, negative_samples)` on the fly using a Python generator, ensuring memory efficiency (RAM).

## File Structure
* `preprocess.py` - Cleans raw `.txt` files, removes punctuation, and prepares a flat string of text.
* `dataset.py` - Handles vocabulary building, the negative sampling table, and the sliding window generator.
* `model.py` - The core mathematical engine. Contains the `Word2VecSGNS` class with the `train_step` method.
* `train.py` - The main training loop. Iterates over epochs, feeds data, updates weights, and saves the trained matrices (`.npy`) and vocab (`.json`) to disk.
* `eval.py` / `play.py` - Inference scripts using Cosine Similarity to find nearest neighbors in the trained vector space.

## How to Run

1. Install the required dependency:
   ```bash
   pip install -r requirements.txt

```

2. (Optional) Run preprocessing if adding new raw `.txt` files to the `data/` directory:
```bash
python preprocess.py

```


3. Start the training loop:
```bash
python train.py

```


4. Test the trained embeddings interactively:
```bash
python play.py

```



## Sample Results

Even with a very small dataset (just three books) and a few epochs of training, the model successfully captures semantic relationships and the specific context of Hemingway's universe.

Here are some actual Nearest Neighbor results (Cosine Similarity) from the trained model:

**Target: 'sea'**

* steam (0.7331)
* waves (0.7256)
* fifes (0.7126)
* heather (0.7109)
* ocean (0.7068)

**Target: 'strong'**

* brave (0.7190)
* gentle (0.7091)
* exciting (0.6781)
* complicated (0.6764)
* strangely (0.6663)

**Target: 'sun'**

* brightness (0.7887)
* shadow (0.7378)
* moonglight (0.7360)
* thunder (0.7355)
* punch (0.7299)

