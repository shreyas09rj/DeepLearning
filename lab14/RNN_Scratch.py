#!/usr/bin/python

# RNN scratch
# next character prediction

import numpy as np

# training data

text = "abcdefghijklmnopqrstuvwxyz"

# create vocabulary
vocab = sorted(set(text))
vocab_size = len(vocab)

# character integer mappings
char_to_ix = {ch: i for i , ch in enumerate(vocab)}
ix_to_char = {i: ch for i , ch in enumerate(vocab)}

print("Vocabulary :",vocab)
print("Vocabulary Size:",vocab_size)

def one_hot(chars):
    x = np.zeros((len(chars), vocab_size))

    for t, ch in enumerate(chars):
        x[t,char_to_ix[ch]] = 1
    return x

#Create input and target seq

seq_len = 10

X_chars = []
Y_chars = []

for i in range(len(text) - seq_len):
    input_seq = text[i:i + seq_len]
    targeted_seq = text[i + 1:i + seq_len + 1]

    X_chars.append(input_seq)
    Y_chars.append(targeted_seq)


X = [one_hot(seq) for seq in X_chars]
Y = [np.array([char_to_ix[ch] for ch in seq],dtype=int)for seq in Y_chars]



print("Number of sequences :",len(X))
print("Number of inputs :",X_chars[0])
print("Number of targets :",Y_chars[0])


class VanillaRNN:
    def __init__(self,input_size,hidden_size,output_size,learning_rate=0.1,seed=42):
        rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.Wxh = rng.normal(0,0.01,(hidden_size,input_size))
        self.Whh = rng.normal(0,0.01,(hidden_size,hidden_size))
        self.Why = rng.normal(0,0.01,(output_size,hidden_size))
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)

    @staticmethod
    def softmax(z):
        z = z - np.max(z)
        exp_z = np.exp(z)
        return ( exp_z / np.sum(exp_z))

#     forward propogation
    def forward(self, xs):
        T = xs.shape[0]

        hs = np.zeros((T + 1,self.hidden_size))
        ys = np.zeros((T,self.Why.shape[0]))
        for t in range(T):
            x_t = xs[t]
            hs[t + 1] = np.tanh(self.Wxh @ x_t+ self.Whh @ hs[t]+ self.bh)
            logits = self.Why @ hs[t + 1] + self.by
            ys[t] = self.softmax(logits)
        return hs, ys

    def loss(self, ys,targets):
        epsilon = 1e-12
        loss = -np.mean(np.log(ys[np.arange(len(targets)),targets] + epsilon ))
        return loss
    def backward(self,xs,hs,ys,targets):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros(self.hidden_size)
        T = xs.shape[0]

        for t in reversed(range(T)):
            dy = ys[t].copy()
            dy[targets[t]] -= 1
            dy /= T
            dWhy += np.outer(dy , hs[t + 1])

            dby += dy
            dh = (self.Why.T @ dy + dh_next)
            da = (1 - hs[t + 1] ** 2)  * dh

            dbh += da
            dWxh += np.outer(da,xs[t])
            dWhh += np.outer(da,hs[t])
            dh_next = (self.Whh.T @ da)
        for gradient in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(gradient, -5, 5, out=gradient)
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

    def train(self, X, Y, epochs=3000,print_every=500):
        for epochs in range(1, epochs + 1):
                total_loss = 0
                for xs ,targets in zip(X, Y):
                    hs, ys = self.forward(xs)
                    current_loss = self.loss(ys,targets)
                    total_loss += (current_loss)
                    self.backward(xs,hs,ys,targets)
                average_loss = (total_loss / len(X))
                if ( epochs == 1 or epochs % print_every == 0):
                    print(f"Epoch {epochs:4d} "
                        f"Loss : {average_loss:.6f} ")

    def predict(self,chars):
        xs = one_hot(chars)
        hs, ys = self.forward(xs)
        predicted_indices = np.argmax(ys, axis=1)
        prediction = ''.join(ix_to_char[i] for i in predicted_indices)
        return prediction

rnn = VanillaRNN(input_size=vocab_size,hidden_size=32,output_size=vocab_size,learning_rate=0.1,seed=42)
print("Starting training")
rnn.train(X,Y,epochs=3000,print_every=500)

print("Finished training")
print("Testing")
test_sequences= ["abcdefghij",
                 "fghijklmno",
                 "klmnopqrst"
                 ]
for sequence in test_sequences:
    prediction = rnn.predict(sequence)
    print("Input :",sequence)
    print("Predicted :",prediction)
    print()

