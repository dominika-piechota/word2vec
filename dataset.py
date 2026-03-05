import numpy as np
from collections import Counter
from pathlib import Path

class Word2VecDataset:
    def __init__(self, input_folder, window_size=2, min_count=3):
        self.window_size = window_size
        self.min_count = min_count
        
        self.word2id = {}
        self.id2word = {}
        self.word_counts = {}
        self.vocab_size = 0
        # TODO: consider implementing subsampling of frequent words (Mikolov 2013) to speed up training

        print("Reading and processing text files...")
        raw_words = self.read_from_folder(input_folder)
        
        print("Making dictionary...")
        self.data_id = self.create_dictionary(raw_words)
        
        print("Preparing table for negative sampling...")
        self.negative_table = self.build_negative_table()

    def read_from_folder(self, folder):
        words = []
        for file in Path(folder).glob('*.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                words.extend(f.read().split())
        return words

    def create_dictionary(self, words):
        cnt = Counter(words)

        sorted_words = [word for word, num in cnt.items() if num >= self.min_count]
        self.vocab_size = len(sorted_words)
        
        # each word gets a unique ID
        self.word2id = {word: i for i, word in enumerate(sorted_words)}
        self.id2word = {i: word for i, word in enumerate(sorted_words)}
        
        # keep track of how many times each word appears (for negative sampling)
        self.word_counts = {self.word2id[word]: cnt[word] for word in sorted_words}
        
        # change the text into a list of IDs. If a word was rejected (because it was rare) we simply skip it
        token_ids = [self.word2id[word] for word in words if word in self.word2id]
        
        print(f"Total number of words in the dataset: {len(token_ids)}")
        print(f"Vocabulary size (unique words): {self.vocab_size}")
        return token_ids

    def build_negative_table(self, table_size=1000000, power=0.75):
        # instead of calculating probabilities every time, we create a large table (1 million elements)
        # where word IDs appear more frequently based on their probability
        # this way, sampling is just picking a random index from this table
        num_of_occurences = np.array([self.word_counts[i] for i in range(self.vocab_size)])
        
        # paper suggests power=0.75 for neg sampling distribution
        weights = num_of_occurences ** power
        probability = weights / np.sum(weights)
        
        # calculate how many times each ID should appear in a table of size 1 million
        occurences_in_table = np.round(probability * table_size).astype(int)
        
        table = []
        for word_id, num in enumerate(occurences_in_table):
            table.extend([word_id] * num)
            
        return np.array(table)

    def draw_neg_samples(self, num_of_samples):
        ids = np.random.randint(0, len(self.negative_table), num_of_samples)
        return self.negative_table[ids]

    def generate_pairs(self, num_neg_samples=5):
        # generator which uses the 'yield' keyword
        # it slides a window over the text and yields data packages

        for i in range(len(self.data_id)):
            center_id = self.data_id[i]
            
            start = max(0, i - self.window_size)
            end = min(len(self.data_id), i + self.window_size + 1)
            
            for j in range(start, end):
                if i == j:
                    continue
                    
                context_id = self.data_id[j]
                
                negative_ids = self.draw_neg_samples(num_neg_samples)
                
                yield center_id, context_id, negative_ids