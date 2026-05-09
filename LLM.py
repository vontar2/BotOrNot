__Author__ = "Eliran Fainberg"


import numpy as np
import re
import os
import time
import string
from collections import Counter

class LLM:
    MAX_SEQUENCE_LENGTH = 512
    EPSILON = 1e-10

    def __init__(self, vocab_size: int, embedding_dim: int, weights_path: str):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights_path = weights_path

        self.tokenizer = Tokenizer()
        self.positional_embedding_layer = PositionalEncoding(LLM.MAX_SEQUENCE_LENGTH, self.embedding_dim, weights_path)
        self.embedding_layer = EmbeddingLayer(self.vocab_size, self.embedding_dim, weights_path, self.positional_embedding_layer)
        self.embedding_layer.load_embeddings()

        self.transformer_block = TransformerBlock(self.embedding_dim, weights_path, self.embedding_layer.embeddings)

        shapes = {
            "embeddings" : self.embedding_layer.embeddings.shape,
            "positional_embeddings" : self.positional_embedding_layer.embeddings.shape,
            "Wq_heads" : self.transformer_block.multi_head_attention[0].Wq_heads[0].shape,
            "Wk_heads": self.transformer_block.multi_head_attention[0].Wk_heads[0].shape,
            "Wv_heads": self.transformer_block.multi_head_attention[0].Wv_heads[0].shape,
            "Wo": self.transformer_block.multi_head_attention[0].Wo.shape,
            "output_weights" : self.transformer_block.mlp_layer[0].output_layer.weights.shape,
            "output_biases" : self.transformer_block.mlp_layer[0].output_layer.biases.shape,
            "hidden_weights" : self.transformer_block.mlp_layer[0].hidden_layers[0].weights.shape,
            "hidden_biases" : self.transformer_block.mlp_layer[0].hidden_layers[0].biases.shape
        }

        self.adam = Adam(shapes, self.transformer_block.layers, self.transformer_block.head_count)


    def forward_pass(self, sentence: str, temperature=0.4):
        if not self.tokenizer.has_vocabulary():
            raise RuntimeError("Vocabulary not initialized. Call create_vocabulary() first.")

        tokens = self.tokenizer.encode(sentence)

        if len(tokens) > LLM.MAX_SEQUENCE_LENGTH:
            tokens = tokens[:LLM.MAX_SEQUENCE_LENGTH]

        embedded_tokens = self.embedding_layer.forward(tokens)
        probabilities = self.transformer_block.forward(embedded_tokens)

        logits = np.log(probabilities[-1] + LLM.EPSILON) / temperature
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        prediction = np.random.choice(len(probs), p=probs)

        return prediction


    @staticmethod
    def cross_entropy_loss(probs, target):
        return -np.log(probs[-1, target] + LLM.EPSILON)


    @staticmethod
    def softmax_gradient(probs, targets):
        """
        Gets ∂L/∂logit_j
        Using  = ∂L/∂logit_j = p_j − 1 while j = target

        :param probs: probabilities output from forward pass
        :param targets: correct tokens
        :return: the overall loss and the loss gradient with respect to the gradient.
        """
        loss = -np.mean(np.log(probs[np.arange(len(targets)), targets] + LLM.EPSILON))

        derivative_logits = probs.copy()
        derivative_logits[np.arange(len(targets)), targets] -= 1
        derivative_logits /= len(targets)

        return loss, derivative_logits


    @staticmethod
    def MLP_layer_normalization_gradient(derivative_out, x):
        """
        Gets the gradient w.r.t. the input using
        ∂L/∂x = ∂L/∂x̂ / √(σ²+ε) + ∂L/∂σ² · 2(x−μ)/N + ∂L/∂μ / N

        while mu = mean, var = variance, x_hat = normalized input
        w.r.t. var:
        ∂L/∂σ² = Σ ∂L/∂x̂_i · (x_i − μ) · −½(σ²+ε)^(−3/2)

        w.r.t. mean:
        ∂L/∂μ = Σ −∂L/∂x̂_i / √(σ²+ε) + ∂L/∂σ² · mean(−2(x−μ))

        :param derivative_out: prev output
        :param x: input (pre-normalization)
        :return: gradient w.r.t. input
        """
        N = x.shape[-1]
        mu = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + LLM.EPSILON)

        d_x_hat = derivative_out
        d_var = np.sum(d_x_hat * (x - mu) * -0.5 * std_inv ** 3, axis=-1, keepdims=True)
        d_mu = np.sum(-d_x_hat * std_inv, axis=-1, keepdims=True) + d_var * np.mean(-2.0 * (x - mu), axis=-1, keepdims=True)
        d_x = d_x_hat * std_inv + d_var * 2.0 * (x - mu) / N + d_mu / N
        return d_x


    @staticmethod
    def relu_backward(d_out, pre_activation):
        """
        Reverse relu using its derivative
        ∂L/∂x = ∂L/∂y · 1[x > 0]
        :param d_out: upstream gradient
        :param pre_activation: pre-activation values (before ReLU was applied)
        :return: relu backwards
        """
        return d_out * (pre_activation > 0).astype(float)


    @staticmethod
    def softmax_backward(d_out, softmax_output):
        """
        Reverse softmax using its jacobian
        ∂L/∂x_i = s_i · (∂L/∂s_i − Σ_j s_j · ∂L/∂s_j)

        :param d_out: upstream gradient
        :param softmax_output: cached softmax output from forward pass
        :return: gradient w.r.t. softmax input
        """
        dot = np.sum(softmax_output * d_out, axis=-1, keepdims=True)
        return softmax_output * (d_out - dot)


    @staticmethod
    def mlp_backward(mlp, d_out: np.ndarray, x_in: np.ndarray):
        """
        MLP backward pass. Uses the pre-activation values cached during the
        forward pass to correctly gate the ReLU gradient — avoids re-running
        forward which would produce a different ReLU mask.

        Gradients:
            ∂L/∂W = Xᵀ @ ∂L/∂Y
            ∂L/∂b = Σ ∂L/∂Y (sum over sequence positions)
            ∂L/∂X = ∂L/∂Y @ Wᵀ

        :param mlp: a MultilayerPerceptron instance
        :param d_out: upstream gradient (shape: [seq_len, output_size])
        :param x_in: input that was fed into the MLP during the forward pass
        :return:
            d_input: gradient w.r.t. MLP input
            grads: dict of weight/bias gradients
        """
        hidden_layer = mlp.hidden_layers[0]
        output_layer = mlp.output_layer

        pre_h = np.dot(x_in, hidden_layer.weights) + hidden_layer.biases
        h = np.maximum(0, pre_h)

        pre_o = np.dot(h, output_layer.weights) + output_layer.biases

        d_pre_o = d_out
        d_W_out = h.T @ d_pre_o
        d_b_out = np.sum(d_pre_o, axis=0, keepdims=True)
        d_h     = d_pre_o @ output_layer.weights.T

        d_pre_h = LLM.relu_backward(d_h, pre_h)
        d_W_hid = x_in.T @ d_pre_h
        d_b_hid = np.sum(d_pre_h, axis=0, keepdims=True)
        d_input = d_pre_h @ hidden_layer.weights.T

        grads = {
            "W_hid": d_W_hid, "b_hid": d_b_hid,
            "W_out": d_W_out, "b_out": d_b_out,
        }
        return d_input, grads


    @staticmethod
    def attention_head_backward(d_out: np.ndarray, Q, K, V, attn_weights: np.ndarray):
        """
        Backward through a single attention head.

        Forward was:
            scores  = Q @ Kᵀ / √d   (+ causal mask)
            A       = softmax(scores)
            out     = A @ V

        Backward:
            ∂L/∂V      = Aᵀ @ ∂L/∂out
            ∂L/∂A      = ∂L/∂out @ Vᵀ
            ∂L/∂scores = softmax_backward(∂L/∂A) / √d
            ∂L/∂Q      = ∂L/∂scores @ K
            ∂L/∂K      = ∂L/∂scoresᵀ @ Q

        :param d_out: upstream gradient (shape: [seq, head_dim])
        :param Q: query projections used in forward (shape: [seq, head_dim])
        :param K: key projections used in forward
        :param V: value projections used in forward
        :param attn_weights: cached softmax output A from forward (shape: [seq, seq])
        :return: (d_Q, d_K, d_V)
        """
        head_dim = K.shape[1]
        scale = 1.0 / np.sqrt(head_dim)

        d_V = attn_weights.T @ d_out
        d_attn_weights = d_out @ V.T

        d_scores = LLM.softmax_backward(d_attn_weights, attn_weights)
        d_scores *= scale

        d_Q = d_scores   @ K
        d_K = d_scores.T @ Q

        return d_Q, d_K, d_V


    @staticmethod
    def multi_head_attention_backward(mha, d_out: np.ndarray, embeddings: np.ndarray):
        """
        Full backward pass through multi-head attention.

        The forward pass was:
            for each head i:
                Q_i = embeddings @ Wq_heads[i]
                K_i = embeddings @ Wk_heads[i]
                V_i = embeddings @ Wv_heads[i]
                head_out_i = softmax(Q_i @ K_iᵀ / √d, causal_mask) @ V_i
            concat  = concat(head_out_0, ..., head_out_n,  axis=1)
            out     = concat @ Wo

        :param mha: the MultiHeadAttention layer (holds Wq_heads, Wk_heads, Wv_heads, Wo)
        :param d_out: upstream gradient (shape: [seq, embedding_dim])
        :param embeddings: normalized input fed into MHA during forward (shape: [seq, embedding_dim])
        :return:
            d_embeddings: gradient w.r.t. the MHA input
            grads: dict with keys Wo, Wq_heads, Wk_heads, Wv_heads
        """
        head_count = len(mha.Wq_heads)
        head_dim   = mha.head_dim

        head_outputs = []
        head_cache   = []

        for i in range(head_count):
            Q = embeddings @ mha.Wq_heads[i]
            K = embeddings @ mha.Wk_heads[i]
            V = embeddings @ mha.Wv_heads[i]

            scores = (Q @ K.T) / np.sqrt(head_dim)

            mask = np.triu(np.ones(scores.shape), k=1).astype(bool)
            scores[mask] = -1e9

            exp_s  = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_w = np.clip(exp_s / np.sum(exp_s, axis=-1, keepdims=True), 1e-10, 1.0)

            head_out = attn_w @ V
            head_outputs.append(head_out)
            head_cache.append((Q, K, V, attn_w))

        concat = np.concatenate(head_outputs, axis=1)

        d_Wo     = concat.T @ d_out
        d_concat = d_out @ mha.Wo.T

        d_embeddings = np.zeros_like(embeddings)
        d_Wq_heads, d_Wk_heads, d_Wv_heads = [], [], []

        for i, (Q, K, V, attn_w) in enumerate(head_cache):
            d_head = d_concat[:, i * head_dim:(i + 1) * head_dim]

            d_Q, d_K, d_V = LLM.attention_head_backward(d_head, Q, K, V, attn_w)

            d_Wq_heads.append(embeddings.T @ d_Q)
            d_Wk_heads.append(embeddings.T @ d_K)
            d_Wv_heads.append(embeddings.T @ d_V)

            d_embeddings += d_Q @ mha.Wq_heads[i].T
            d_embeddings += d_K @ mha.Wk_heads[i].T
            d_embeddings += d_V @ mha.Wv_heads[i].T

        grads = {
            "Wo":       d_Wo,
            "Wq_heads": d_Wq_heads,
            "Wk_heads": d_Wk_heads,
            "Wv_heads": d_Wv_heads,
        }
        return d_embeddings, grads


    def backward_pass(self, tokens: list, target_token_id: list, cycle):
        """
        Full backward pass.

        Key fixes vs. original:
        1.  The residual-stream gradient through a sub-layer is:
                d_x_prev = d_x_next + through_layernorm(d_sublayer_input, pre_norm_input)
            i.e. we ADD the layernorm gradient to the residual coming *from above*,
            not to the sublayer input gradient alone.

        2.  mlp_backward / multi_head_attention_backward now receive the MHA/MLP
            layer objects directly so they reference the correct weight arrays.

        :param cycle: optimizer step count (used for Adam bias correction).
        :param tokens: input token ids.
        :param target_token_id: target (next) token ids.
        :return: scalar loss.
        """
        tb = self.transformer_block

        if len(tokens) > self.MAX_SEQUENCE_LENGTH:
            tokens          = tokens[:self.MAX_SEQUENCE_LENGTH]
            target_token_id = target_token_id[:self.MAX_SEQUENCE_LENGTH]

        x = self.embedding_layer.forward(tokens)

        layer_cache = []

        for i in range(tb.layers):
            x_before_attn = x.copy()

            x_norm_attn = tb.normalize(x_before_attn)
            attn_out    = tb.multi_head_attention[i].forward(x_norm_attn)
            x           = x_before_attn + attn_out

            x_after_attn = x.copy()

            x_norm_mlp = tb.normalize(x_after_attn)
            mlp_out    = tb.mlp_layer[i].forward(x_norm_mlp)
            x          = x_after_attn + mlp_out

            layer_cache.append({
                "x_before_attn": x_before_attn,
                "x_norm_attn":   x_norm_attn,
                "x_after_attn":  x_after_attn,
                "x_norm_mlp":    x_norm_mlp,
            })

        x_final = x

        logits = x_final @ tb.w_model_head
        probs  = tb.softmax.forward(logits)

        loss, d_logits = LLM.softmax_gradient(probs, target_token_id)

        d_x = d_logits @ tb.w_model_head.T

        d_emb_tied = x_final.T @ d_logits
        all_mha_grads = []
        all_mlp_grads = []

        for i in reversed(range(tb.layers)):
            cache = layer_cache[i]

            d_mlp_norm_out, mlp_grads = LLM.mlp_backward(
                tb.mlp_layer[i], d_x, cache["x_norm_mlp"]
            )
            d_mlp_prenorm = LLM.MLP_layer_normalization_gradient(
                d_mlp_norm_out, cache["x_after_attn"]
            )
            d_x_after_attn = d_x + d_mlp_prenorm

            d_mha_norm_out, mha_grads = LLM.multi_head_attention_backward(
                tb.multi_head_attention[i], d_x_after_attn, cache["x_norm_attn"]
            )
            d_mha_prenorm = LLM.MLP_layer_normalization_gradient(
                d_mha_norm_out, cache["x_before_attn"]
            )
            d_x = d_x_after_attn + d_mha_prenorm

            all_mha_grads.insert(0, mha_grads)
            all_mlp_grads.insert(0, mlp_grads)

        d_token_emb = d_x

        d_embeddings_table = np.zeros_like(self.embedding_layer.embeddings)
        for j, tok_id in enumerate(tokens):
            d_embeddings_table[tok_id] += d_token_emb[j]
        d_embeddings_table += d_emb_tied.T

        d_pos_embeddings = np.zeros_like(self.positional_embedding_layer.embeddings)
        for j in range(len(tokens)):
            d_pos_embeddings[j] += d_token_emb[j]

        e_to_sub, pe_to_sub = self.adam.optimize_embeddings(
            d_embeddings_table, d_pos_embeddings, cycle
        )
        self.embedding_layer.embeddings              -= e_to_sub
        self.positional_embedding_layer.embeddings   -= pe_to_sub

        for i in range(tb.layers):
            mha = tb.multi_head_attention[i]
            mg  = all_mha_grads[i]
            mlp = tb.mlp_layer[i]
            pg  = all_mlp_grads[i]

            mha.Wo -= self.adam.optimize_Wo(mg["Wo"], i, cycle)

            for h in range(len(mha.Wq_heads)):
                Wq_sub, Wk_sub, Wv_sub = self.adam.optimize_head(
                    mg["Wq_heads"][h], mg["Wk_heads"][h], mg["Wv_heads"][h],
                    cycle, i, h
                )
                mha.Wq_heads[h] -= Wq_sub
                mha.Wk_heads[h] -= Wk_sub
                mha.Wv_heads[h] -= Wv_sub

            W_hid_sub, b_hid_sub, W_out_sub, b_out_sub = self.adam.optimize_mlp(
                pg["W_hid"], pg["b_hid"], pg["W_out"], pg["b_out"], cycle, i
            )
            mlp.hidden_layers[0].weights -= W_hid_sub
            mlp.hidden_layers[0].biases  -= b_hid_sub
            mlp.output_layer.weights     -= W_out_sub
            mlp.output_layer.biases      -= b_out_sub

        return loss


    def train(self, corpus: list, epochs: int = 10):
        counter = 1
        for epoch in range(epochs):
            total_loss = 0.0
            epoch_start = time.time()

            for tokens in corpus:
                if len(tokens) < 2:
                    continue

                inputs  = tokens[:-1]
                targets = tokens[1:]

                loss = self.backward_pass(inputs, targets, counter)
                total_loss += loss
                counter += 1

                if counter % 500 == 0:
                    elapsed = time.time() - epoch_start
                    print(f"  [{counter / (epoch + 1)}/{len(corpus)}] loss: {loss:.4f} | {elapsed:.0f}s elapsed")

            print(f"Epoch {epoch + 1}/{epochs} | loss: {total_loss / len(corpus):.4f}")

            if epoch % 2 == 0:
                self.save_model()
                print("Model saved")

        self.save_model()
        print("Model saved")


    def save_model(self):
        before_pos = time.time()
        self.positional_embedding_layer.save_embeddings()
        print(f"Positional embeddings saved after {time.time() - before_pos}")

        before_embeddings = time.time()
        self.embedding_layer.save_embeddings()
        print(f"Embeddings saved after {time.time() - before_embeddings}")

        before_tb = time.time()
        self.transformer_block.save_transformer_block()
        print(f"Transformer saved after {time.time() - before_tb}")


    def load_model(self):
        self.positional_embedding_layer.load_embeddings()
        self.embedding_layer.load_embeddings()
        self.transformer_block = TransformerBlock(self.embedding_dim, self.weights_path, self.embedding_layer.embeddings)
        self.transformer_block.load_transformer_block()


    def create_vocabulary(self, text):
        splits = self.tokenizer.get_splits_for_training(text)
        self.tokenizer.BPE(splits, self.vocab_size)


    def decode_prediction(self, prediction):
        return self.tokenizer.decode([int(prediction)])

    def test(self, sentences: list):
        for idx, tokens in enumerate(sentences):
            if len(tokens) < 2:
                continue

            print(f"\n--- Sentence {idx + 1} ---")
            predicted_tokens = []

            for i in range(len(tokens) - 1):
                prediction = self.forward_pass(self.tokenizer.decode(tokens[:i + 1]))
                predicted_tokens.append(prediction)

                pred_str = self.tokenizer.decode([prediction])
                target_str = self.tokenizer.decode([tokens[i + 1]])
                correct = "✓" if prediction == tokens[i + 1] else "✗"
                print(f"  {correct} predicted={repr(pred_str):15s}  target={repr(target_str)}")

            print(f"  output: {self.tokenizer.decode(predicted_tokens)}")


class Adam:
    def __init__(self, shapes, t_count, h_count, learning_rate=0.00003, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.v_embeddings     = np.zeros(shapes["embeddings"])
        self.v_pos_embeddings = np.zeros(shapes["positional_embeddings"])
        self.v_Wq_heads       = [[np.zeros(shapes["Wq_heads"]) for _ in range(h_count)] for _ in range(t_count)]
        self.v_Wk_heads       = [[np.zeros(shapes["Wk_heads"]) for _ in range(h_count)] for _ in range(t_count)]
        self.v_Wv_heads       = [[np.zeros(shapes["Wv_heads"]) for _ in range(h_count)] for _ in range(t_count)]
        self.v_Wo             = [np.zeros(shapes["Wo"]) for _ in range(t_count)]
        self.v_output_weights = [np.zeros(shapes["output_weights"]) for _ in range(t_count)]
        self.v_output_biases  = [np.zeros(shapes["output_biases"]) for _ in range(t_count)]
        self.v_hidden_weights = [np.zeros(shapes["hidden_weights"]) for _ in range(t_count)]
        self.v_hidden_biases  = [np.zeros(shapes["hidden_biases"]) for _ in range(t_count)]

        self.m_embeddings     = np.zeros(shapes["embeddings"])
        self.m_pos_embeddings = np.zeros(shapes["positional_embeddings"])
        self.m_Wq_heads       = [[np.zeros(shapes["Wq_heads"]) for _ in range(h_count)] for _ in range(t_count)]
        self.m_Wk_heads       = [[np.zeros(shapes["Wk_heads"]) for _ in range(h_count)] for _ in range(t_count)]
        self.m_Wv_heads       = [[np.zeros(shapes["Wv_heads"]) for _ in range(h_count)] for _ in range(t_count)]
        self.m_Wo             = [np.zeros(shapes["Wo"]) for _ in range(t_count)]
        self.m_output_weights = [np.zeros(shapes["output_weights"]) for _ in range(t_count)]
        self.m_output_biases  = [np.zeros(shapes["output_biases"]) for _ in range(t_count)]
        self.m_hidden_weights = [np.zeros(shapes["hidden_weights"]) for _ in range(t_count)]
        self.m_hidden_biases  = [np.zeros(shapes["hidden_biases"]) for _ in range(t_count)]

        self.b1 = beta1
        self.b2 = beta2
        self.epsilon = epsilon
        self.lr      = learning_rate


    def _update(self, m, v, g, t):
        """Single Adam update step. Returns (new_m, new_v, delta_to_subtract)."""
        new_m    = self.b1 * m + (1 - self.b1) * g
        new_v    = self.b2 * v + (1 - self.b2) * (g ** 2)
        m_hat    = new_m / (1 - self.b1 ** t)
        v_hat    = new_v / (1 - self.b2 ** t)
        to_sub   = (self.lr * m_hat) / (v_hat ** 0.5 + self.epsilon)
        return new_m, new_v, to_sub


    def optimize_embeddings(self, g_embeddings, g_positions, t):
        self.m_embeddings, self.v_embeddings, e_to_sub = self._update(
            self.m_embeddings, self.v_embeddings, g_embeddings, t)
        self.m_pos_embeddings, self.v_pos_embeddings, pe_to_sub = self._update(
            self.m_pos_embeddings, self.v_pos_embeddings, g_positions, t)
        return e_to_sub, pe_to_sub


    def optimize_Wo(self, g_o, t_id, t):
        self.m_Wo[t_id], self.v_Wo[t_id], to_sub = self._update(
            self.m_Wo[t_id], self.v_Wo[t_id], g_o, t)
        return to_sub


    def optimize_head(self, g_q, g_k, g_v, t, t_id, h_id):
        self.m_Wq_heads[t_id][h_id], self.v_Wq_heads[t_id][h_id], q_sub = self._update(
            self.m_Wq_heads[t_id][h_id], self.v_Wq_heads[t_id][h_id], g_q, t)
        self.m_Wk_heads[t_id][h_id], self.v_Wk_heads[t_id][h_id], k_sub = self._update(
            self.m_Wk_heads[t_id][h_id], self.v_Wk_heads[t_id][h_id], g_k, t)
        self.m_Wv_heads[t_id][h_id], self.v_Wv_heads[t_id][h_id], v_sub = self._update(
            self.m_Wv_heads[t_id][h_id], self.v_Wv_heads[t_id][h_id], g_v, t)
        return q_sub, k_sub, v_sub


    def optimize_mlp(self, g_W_hid, g_b_hid, g_W_out, g_b_out, t, mlp_id):
        self.m_hidden_weights[mlp_id], self.v_hidden_weights[mlp_id], W_hid_sub = self._update(
            self.m_hidden_weights[mlp_id], self.v_hidden_weights[mlp_id], g_W_hid, t)
        self.m_hidden_biases[mlp_id], self.v_hidden_biases[mlp_id], b_hid_sub = self._update(
            self.m_hidden_biases[mlp_id], self.v_hidden_biases[mlp_id], g_b_hid, t)
        self.m_output_weights[mlp_id], self.v_output_weights[mlp_id], W_out_sub = self._update(
            self.m_output_weights[mlp_id], self.v_output_weights[mlp_id], g_W_out, t)
        self.m_output_biases[mlp_id], self.v_output_biases[mlp_id], b_out_sub = self._update(
            self.m_output_biases[mlp_id], self.v_output_biases[mlp_id], g_b_out, t)
        return W_hid_sub, b_hid_sub, W_out_sub, b_out_sub


class Tokenizer:
    PATH = "tokens_weights.txt"
    def __init__(self):
        self.path           = Tokenizer.PATH
        self.pattern        = r"<BOS>|<EOS>|<USER>|<BOT>| ?[\w']+|[^\w\s]"
        self.SPECIAL_TOKENS = ["<PAD>", "<EOS>", "<UNK>", "<BOS>", "<USER>", "<BOT>"]
        self.basic          = list(string.ascii_letters) + list(string.digits) + list(string.punctuation)

        if os.path.isfile(self.path):
            with open(self.path, 'r') as f:
                self.vocab = list(self.SPECIAL_TOKENS + f.read().split("\n") + self.basic)
        else:
            self.vocab = self.SPECIAL_TOKENS + self.basic


    def has_vocabulary(self):
        return len(self.vocab) > 0


    def tokenize(self, word: str) -> list:
        if word in self.SPECIAL_TOKENS:
            return [self.vocab.index(word)]

        tokens = []
        left   = 0

        while left < len(word):
            for end in range(len(word), left, -1):
                if word[left:end] in self.vocab:
                    tokens.append(word[left:end])
                    left = end
                    break
            else:
                tokens.append(word[left])
                left += 1

        indices = []
        for token in tokens:
            if token in self.vocab:
                indices.append(self.vocab.index(token))
            else:
                indices.append(self.vocab.index("<UNK>"))

        return indices


    def get_splits_for_training(self, text: str) -> list:
        text = re.findall(self.pattern, text)
        text = [list(word) for word in text]
        return text

    def get_splits(self, text: str) -> list:
        return re.findall(self.pattern, text)

    def BPE(self, splits: list, desired_vocab_size: int) -> None:
        pairs_frequencies = Counter()
        for word in splits:
            for a, b in zip(word[:-1], word[1:]):
                pairs_frequencies[a + b] += 1

        while len(self.vocab) < desired_vocab_size:
            if not pairs_frequencies:
                break
            if len(self.vocab) % 100 == 0:
                print(len(self.vocab))

            max_pair = pairs_frequencies.most_common(1)[0][0]

            if max_pair not in self.vocab:
                self.vocab.append(max_pair)

            for word in splits:
                i = 0
                while i < len(word) - 1:
                    if word[i] + word[i + 1] == max_pair:
                        if i > 0:
                            pairs_frequencies[word[i - 1] + word[i]] -= 1
                        if i + 2 < len(word):
                            pairs_frequencies[word[i + 1] + word[i + 2]] -= 1

                        word[i] += word[i + 1]
                        del word[i + 1]

                        if i > 0:
                            pairs_frequencies[word[i - 1] + word[i]] += 1
                        if i + 1 < len(word):
                            pairs_frequencies[word[i] + word[i + 1]] += 1
                    else:
                        i += 1

            del pairs_frequencies[max_pair]

        self.save_vocab()


    def save_vocab(self):
        with open(self.path, 'w') as f:
            f.write("\n".join(self.vocab))

    def encode(self, text: str) -> list:
        splits  = self.get_splits(text)
        tokens  = []
        for word in splits:
            tokens.extend(self.tokenize(word))
        return tokens

    def decode(self, ids: list) -> str:
        if len(ids) == 1:
            return self.vocab[ids[0]]
        return "".join([self.vocab[_id] for _id in ids])


class EmbeddingLayer:
    def __init__(self, vocab_size: int, embedding_dim: int, weights_path: str, positional_embeddings_layer):
        self.vocab_size                  = vocab_size
        self.embedding_dim               = embedding_dim
        self.weights_path                = weights_path
        self.embeddings                  = None
        self.positional_embeddings_layer = positional_embeddings_layer

        self.positional_embeddings_layer.load_embeddings()
        self.positional_embeddings = self.positional_embeddings_layer.embeddings

    def init_weights(self):
        self.embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.02

    def forward(self, token_ids: list) -> np.ndarray:
        x = self.embeddings[token_ids]
        x = x + self.positional_embeddings_layer.forward(len(token_ids))
        return x

    def save_embeddings(self):
        np.savez(self.weights_path, embeddings=self.embeddings)

    def load_embeddings(self):
        if os.path.exists(self.weights_path):
            data            = np.load(self.weights_path, allow_pickle=True)
            self.embeddings = data["embeddings"]
        else:
            self.init_weights()


class PositionalEncoding:
    def __init__(self, max_context_length: int, embedding_dim: int, weights_path: str):
        self.max_context_length = max_context_length
        self.embedding_dim      = embedding_dim
        self.weights_path       = "pos_" + weights_path
        self.embeddings         = None

    def init_weights(self):
        self.embeddings = np.random.randn(self.max_context_length, self.embedding_dim) * 0.02

    def forward(self, n: int) -> np.ndarray:
        return self.embeddings[np.arange(n)]

    def save_embeddings(self):
        np.savez(self.weights_path, pos_embeddings=self.embeddings)

    def load_embeddings(self):
        if os.path.exists(self.weights_path):
            data            = np.load(self.weights_path, allow_pickle=True)
            self.embeddings = data["pos_embeddings"]
        else:
            self.init_weights()


class TransformerBlock:
    def __init__(self, embedding_dim: int, weights_path: str, embeddings: np.ndarray, head_count=8, layers=6):
        self.mlp_path       = "mlp_"       + weights_path
        self.attention_path = "attention_" + weights_path

        self.multi_head_attention = [MultiHeadAttention(embedding_dim, head_count=head_count) for _ in range(layers)]

        self.layers     = layers
        self.head_count = head_count
        self.mlp_layer  = [MLP(1, 256, 256) for _ in range(layers)]
        self.softmax    = Softmax()

        self.w_model_head = embeddings.T

    def forward(self, seq_embs) -> np.ndarray:
        for i in range(self.layers):
            seq_embs = seq_embs + self.multi_head_attention[i].forward(self.normalize(seq_embs))
            seq_embs = seq_embs + self.mlp_layer[i].forward(self.normalize(seq_embs))

        logits = seq_embs @ self.w_model_head
        return self.softmax.forward(logits)

    def save_transformer_block(self):
        self.save_mlp_weights()
        self.save_attention_weights()

    def load_transformer_block(self):
        self.load_mlp_weights()
        self.load_attention_weights()

    def save_mlp_weights(self):
        output_weights = self.mlp_layer[0].output_layer.weights
        output_biases  = self.mlp_layer[0].output_layer.biases
        weights        = self.mlp_layer[0].hidden_layers[0].weights
        biases         = self.mlp_layer[0].hidden_layers[0].biases

        for mlp in self.mlp_layer[1:]:
            weights        = np.concatenate((weights,        mlp.hidden_layers[0].weights), axis=0)
            biases         = np.concatenate((biases,         mlp.hidden_layers[0].biases),  axis=0)
            output_weights = np.concatenate((output_weights, mlp.output_layer.weights),     axis=0)
            output_biases  = np.concatenate((output_biases,  mlp.output_layer.biases),      axis=0)

        np.savez(self.mlp_path, weights=weights, biases=biases,
                 output_weights=output_weights, output_biases=output_biases, allow_pickle=True)

    def load_mlp_weights(self):
        data           = np.load(self.mlp_path, allow_pickle=True)
        weights        = data["weights"]
        biases         = data["biases"]
        output_weights = data["output_weights"]
        output_biases  = data["output_biases"]

        split_weights        = np.array_split(weights,        self.layers)
        split_biases         = np.array_split(biases,         self.layers)
        split_output_weights = np.array_split(output_weights, self.layers)
        split_output_biases  = np.array_split(output_biases,  self.layers)

        for mlp, _w, _b, _ow, _ob in zip(self.mlp_layer, split_weights, split_biases, split_output_weights, split_output_biases):
            mlp.hidden_layers[0].weights = _w
            mlp.hidden_layers[0].biases  = _b
            mlp.output_layer.weights     = _ow
            mlp.output_layer.biases      = _ob

    def save_attention_weights(self):
        """
        FIX: save/load now work entirely from Wq_heads/Wk_heads/Wv_heads lists
        (the monolithic Wq/Wk/Wv attributes were stale after __init__).
        """
        # Reconstruct monolithic matrices from the head slices
        Wq = np.concatenate(self.multi_head_attention[0].Wq_heads, axis=1)
        Wk = np.concatenate(self.multi_head_attention[0].Wk_heads, axis=1)
        Wv = np.concatenate(self.multi_head_attention[0].Wv_heads, axis=1)
        Wo = self.multi_head_attention[0].Wo

        for mha in self.multi_head_attention[1:]:
            Wq = np.concatenate((Wq, np.concatenate(mha.Wq_heads, axis=1)), axis=0)
            Wk = np.concatenate((Wk, np.concatenate(mha.Wk_heads, axis=1)), axis=0)
            Wv = np.concatenate((Wv, np.concatenate(mha.Wv_heads, axis=1)), axis=0)
            Wo = np.concatenate((Wo, mha.Wo), axis=0)

        np.savez(self.attention_path,
                 Wq_weights=Wq, Wk_weights=Wk, Wv_weights=Wv,
                 Wo_weights=Wo, W_model_head=self.w_model_head,
                 allow_pickle=True)

    def load_attention_weights(self):
        """
        FIX: after loading, re-split into Wq_heads/Wk_heads/Wv_heads so the
        backward pass and forward pass both see the same weight arrays.
        """
        data       = np.load(self.attention_path, allow_pickle=True)
        Wq_weights = data["Wq_weights"]
        Wk_weights = data["Wk_weights"]
        Wv_weights = data["Wv_weights"]
        Wo_weights = data["Wo_weights"]
        self.w_model_head = data["W_model_head"]

        split_Wq = np.array_split(Wq_weights, self.layers)
        split_Wk = np.array_split(Wk_weights, self.layers)
        split_Wv = np.array_split(Wv_weights, self.layers)
        split_Wo = np.array_split(Wo_weights, self.layers)

        for mha, _Wq, _Wk, _Wv, _Wo in zip(self.multi_head_attention, split_Wq, split_Wk, split_Wv, split_Wo):
            mha.Wq_heads = np.array_split(_Wq, self.head_count, axis=1)
            mha.Wk_heads = np.array_split(_Wk, self.head_count, axis=1)
            mha.Wv_heads = np.array_split(_Wv, self.head_count, axis=1)
            mha.Wo       = _Wo

    def normalize(self, seq_embs) -> np.ndarray:
        return (seq_embs - np.mean(seq_embs, axis=-1, keepdims=True)) / \
               np.sqrt(np.var(seq_embs, axis=-1, keepdims=True) + 1e-8)


class MultiHeadAttention:
    def __init__(self, embedding_dim: int, head_count=8):
        self.embedding_dim = embedding_dim
        self.head_dim      = embedding_dim // head_count

        self.Wo = np.random.randn(256, 256) * 0.02
        Wq      = np.random.randn(256, self.embedding_dim) * 0.02
        Wk      = np.random.randn(256, self.embedding_dim) * 0.02
        Wv      = np.random.randn(256, self.embedding_dim) * 0.02

        # Only keep the split-head versions; the monolithic copies are discarded
        self.Wq_heads = np.array_split(Wq, head_count, axis=1)
        self.Wk_heads = np.array_split(Wk, head_count, axis=1)
        self.Wv_heads = np.array_split(Wv, head_count, axis=1)

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        head_outputs = []
        for i in range(len(self.Wq_heads)):
            Q = embeddings @ self.Wq_heads[i]
            K = embeddings @ self.Wk_heads[i]
            V = embeddings @ self.Wv_heads[i]
            head_outputs.append(AttentionHead(Q, K, V).forward())

        return np.concatenate(head_outputs, axis=1) @ self.Wo


class AttentionHead:
    def __init__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray):
        self.softmax = Softmax()
        self.Q = Q
        self.K = K
        self.V = V

    def forward(self) -> np.ndarray:
        scores = np.dot(self.Q, self.K.T) / np.sqrt(np.shape(self.K)[1])
        scores[np.triu(np.ones(scores.shape), k=1).astype(bool)] = -1e9
        return np.dot(self.softmax.forward(scores), self.V)

class MLP:
    HIDDEN_SIZE = 1024

    def __init__(self, hidden_layer_count, output_size, input_size):
        self.hidden_layers = [Layer(input_size, MLP.HIDDEN_SIZE)]
        for _ in range(hidden_layer_count - 1):
            self.hidden_layers.append(Layer(MLP.HIDDEN_SIZE, MLP.HIDDEN_SIZE))
        self.output_layer = Layer(MLP.HIDDEN_SIZE, output_size)

    def forward(self, n_input):
        for layer in self.hidden_layers:
            n_input = layer.forward(n_input)[1]
        return self.output_layer.forward(n_input)[0]


class Layer:
    def __init__(self, n_features, layer_size):
        self.weights          = np.random.randn(n_features, layer_size) * 0.1
        self.biases           = np.random.randn(1, layer_size)
        self.output           = None
        self.activated_output = None

    def forward(self, n_input):
        self.output           = np.dot(n_input, self.weights) + self.biases
        self.activated_output = Relu.forward(self.output)
        return self.output, self.activated_output


class Relu:
    @staticmethod
    def forward(x):
        return np.maximum(0, x)


class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        remove_max = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = np.clip(remove_max / np.sum(remove_max, axis=-1, keepdims=True), 1e-10, 1.0)
        return self.output