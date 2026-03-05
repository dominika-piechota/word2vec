import numpy as np

def get_similar_words(target_word, model, dataset, top_n=5):

    if target_word not in dataset.word2id:
        print(f"Error: Word '{target_word}' is not in the vocabulary")
        return []
        
    word_id = dataset.word2id[target_word]
    word_vector = model.W1[word_id]
    
    word_norm = np.linalg.norm(word_vector)
    
    all_norms = np.linalg.norm(model.W1, axis=1)
    
    dot_products = np.dot(model.W1, word_vector)
    
    similarities = dot_products / (all_norms * word_norm + 1e-10) # deviding by zero protection

    sorted_indices = np.argsort(similarities)[::-1][:top_n + 1]
    
    print(f"\nBest matches for'{target_word}':")
    print("-" * 30)
    
    results = []
    for idx in sorted_indices:
        if idx == word_id:
            continue
            
        similar_word = dataset.id2word[idx]
        score = similarities[idx]
        results.append((similar_word, score))
        
        print(f"{similar_word:15} | result: {score:.4f}")
        
        if len(results) == top_n:
            break
            
    return results