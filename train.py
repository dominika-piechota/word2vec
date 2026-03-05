from dataset import Word2VecDataset
from model import Word2VecSGNS
from eval import get_similar_words
import numpy as np
import time
import json

EMBEDDING_DIM = 50
LEARNING_RATE = 0.05
EPOCHS = 5
NUM_NEG_SAMPLES = 5

print("STEP 1: DATA PREPARATION")
dataset = Word2VecDataset('cleaned_data', window_size=2, min_count=3)

print("\nSTEP 2: MODEL INITIALIZATION")
model = Word2VecSGNS(dataset.vocab_size, EMBEDDING_DIM, LEARNING_RATE)

print("\nSTEP 3: TRAINING")
for epoch in range(EPOCHS):
    total_loss = 0
    step = 0
    start_time = time.time()
    
    for center_id, context_id, negative_ids in dataset.generate_pairs(num_neg_samples=NUM_NEG_SAMPLES):
        loss = model.train_step(center_id, context_id, negative_ids)
        
        total_loss += loss
        step += 1
        
        if step % 100000 == 0:
            avg_loss_so_far = total_loss / step
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {step} | Average loss: {avg_loss_so_far:.4f}")
            
    end_time = time.time()
    epoch_avg_loss = total_loss / max(1, step)
    epoch_time = end_time - start_time
    
    print(f"EPOCH {epoch+1} COMPLETED! Average Loss: {epoch_avg_loss:.4f} | Time: {epoch_time:.2f}s\n")
    
print("Training completed! Word2Vec model is ready.")

print("\nRESULTS TESTING")
words_to_check = ["old", "man", "and", "the", "sea"]

for word in words_to_check:
    get_similar_words(word, model, dataset, top_n=5)

print("\nSAVING THE MODEL")

# saving matrix of weights W1 and W2 as .npy files
np.save('W1_weights.npy', model.W1)
np.save('W2_weights.npy', model.W2)

# saving the vocabulary (word2id mapping) as a JSON file
with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(dataset.word2id, f)

print("Files W1_weights.npy, W2_weights.npy and vocab.json have been saved. You can use them later for loading the model and performing evaluations.")