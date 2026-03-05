import numpy as np
import json
from eval import get_similar_words

class LoadedDataset:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
            
        # we have to convert the keys of id2word back to integers, because JSON saves them as strings
        self.id2word = {int(id_num): word for word, id_num in self.word2id.items()}

class LoadedModel:
    def __init__(self, w1_path):
        # we need only W1 to get the word vectors for evaluation
        self.W1 = np.load(w1_path)

print("Loading the saved model")
dataset = LoadedDataset('vocab.json')
model = LoadedModel('W1_weights.npy')

print("\nEverything is ready - You can now test the model by typing any word from the vocabulary. \nThe program will show you the most similar words based on the trained Word2Vec model.")
print("Enter 'exit' to quit the program.\n")

while True:
    word = input("Your word > ").strip().lower()
    
    if word == 'exit':
        print("Bye! Thanks for testing the model!")
        break
        
    if not word:
        continue
        
    get_similar_words(word, model, dataset, top_n=5)
    print("\n" + "="*40 + "\n")