import numpy as np

def sigmoid(x):
    x = np.clip(x, -10, 10) # don't let x be too large or too small - numerical stability
    return 1.0 / (1.0 + np.exp(-x))

class Word2VecSGNS:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # central word vectors (input)
        self.W1 = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim))
        
        # context word vectors (output)
        self.W2 = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim))

    def train_step(self, center_id, context_id, negative_ids):
        # FORWARD PASS
        v_c = self.W1[center_id]
        u_w = self.W2[context_id]
        u_n = self.W2[negative_ids]

        pos_dot = np.dot(v_c, u_w)
        p_pos = sigmoid(pos_dot)

        neg_dot = np.dot(u_n, v_c)
        p_neg = sigmoid(neg_dot)

        loss_pos = -np.log(p_pos + 1e-10) # add epsilon to prevent log(0)
        loss_neg = -np.sum(np.log(1.0 - p_neg + 1e-10))
        total_loss = loss_pos + loss_neg

        # BACKWARD PASS
        err_pos = p_pos - 1.0
        err_neg = p_neg

        grad_u_w = err_pos * v_c
        # print(e_neg.shape, v_c.shape) # debug
        grad_u_n = err_neg[:, np.newaxis] * v_c
        
        grad_v_c = (err_pos * u_w) + np.sum(err_neg[:, np.newaxis] * u_n, axis=0)

        self.W1[center_id] -= self.learning_rate * grad_v_c
        self.W2[context_id] -= self.learning_rate * grad_u_w
        self.W2[negative_ids] -= self.learning_rate * grad_u_n

        return total_loss