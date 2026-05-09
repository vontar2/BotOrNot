import numpy as np
from LLM import *
from datasets import load_dataset
import pandas as pd
import string

class MLP:
    HIDDEN_SIZE = 3

    def __init__(self, hidden_layer_count, output_size, input_size):
        self.hidden_layers = []
        self.hidden_layers.append(Layer(input_size, MLP.HIDDEN_SIZE))

        for _ in range(hidden_layer_count - 1):
            self.hidden_layers.append(Layer(MLP.HIDDEN_SIZE, MLP.HIDDEN_SIZE))

        self.output_layer = Layer(MLP.HIDDEN_SIZE, output_size)


    def forward(self, n_input):
        for layer in self.hidden_layers:
            n_input = layer.forward(n_input)[1]

        return self.output_layer.forward(n_input)[0]


class Layer:
    def __init__(self, n_features, layer_size):
        self.weights = np.random.randn(n_features, layer_size) * 0.1
        self.biases = np.random.randn(1, layer_size)
        self.output = None
        self.activated_output = None


    def forward(self, n_input):
        self.output = np.dot(n_input, self.weights) + self.biases
        self.activated_output = Relu.forward(self.output)
        return self.output, self.activated_output

class Relu:
    @staticmethod
    def forward(x):
        output =  np.maximum(0, x)
        return output

class Softmax:
    """
    Turns a list of integers into a list of probabilities which sum to 1.
    """
    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        remove_max = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = np.clip(remove_max / np.sum(remove_max, axis=-1, keepdims=True), 1e-10, 1.0)
        return self.output


mlp_layer = []

def save_mlp_weights():
    weights = mlp_layer[0].hidden_layers[0].weights
    biases = mlp_layer[0].hidden_layers[0].biases

    print(weights.shape)
    print(biases.shape)

    for mlp in mlp_layer[1:]:
        weights = np.concatenate((weights, mlp.hidden_layers[0].weights), axis=0)
        biases = np.concatenate((biases, mlp.hidden_layers[0].biases), axis=0)

    np.savez("testSave.npz", weights=np.array(weights), biases=np.array(biases),allow_pickle=True)

def load_mlp_weights():
    weights = np.load("testSave.npz")["weights"]
    biases = np.load("testSave.npz")["biases"]

    split_weights = np.vsplit(weights, 6)
    split_biases = np.vsplit(biases, 6)

    for mlp, _weights, _biases in zip(mlp_layer, split_weights, split_biases):
        mlp.hidden_layers[0].weights = _weights
        mlp.hidden_layers[0].biases = _biases

def test_saves():
    SAVE = False

    if not SAVE:
        load_mlp_weights()
        for i, _mlp in enumerate(mlp_layer):
            print(_mlp.hidden_layers[0].weights)
            print(i)

    else:
        save_mlp_weights()
        for i, _mlp in enumerate(mlp_layer):
            print(_mlp.hidden_layers[0].weights)
            print(i)


def test_vocab():# Option A
    dataset = load_dataset("agentlans/li2017dailydialog")
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"  # beginning of sequence
    EOS_TOKEN = "<EOS>"  # end of sequence
    USR_TOKEN = "<USR>"  # marks human turn
    BOT_TOKEN = "<BOT>"

    # Access splits
    train = dataset["train"]
    val = dataset["validation"]
    test = dataset["test"]

    example = train[0]
    conversation = example["conversations"]

    all_pairs = []

    def format_conversation(_example):
        turns = [m for m in _example["conversations"] if m["from"] != "system"]

        text = ""
        for msg in turns:
            text += msg["value"] + " "


        return text


    def learn_vocab(training):
        text = " "
        for ex in training:
            conv = ex["conversations"]
            turns = [m for m in conv if m["from"] != "system"]
            for msg in turns:
                text += msg["value"] + " "

        llm = LLM(50000, 256, "weights.npz")
        llm.create_vocabulary(text)

    learn_vocab(train)

def test_tokenization():
    sentence = "hello everybody how are you, here's a tokenization for you"

    llm = LLM(50000, 256, "weights.npz")
    ids = llm.tokenizer.tokenize(sentence)
    print(ids)
    print(llm.tokenizer.decode(ids))

    pred = int(llm.forward_pass(sentence))

    ids.append(pred)

    print(llm.tokenizer.decode(ids))

def format_corpus():
    dataset = load_dataset("agentlans/li2017dailydialog")
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"  # beginning of sequence
    EOS_TOKEN = "<EOS>"  # end of sequence
    USR_TOKEN = "<USR>"  # marks human turn
    BOT_TOKEN = "<BOT>"

    # Access splits
    train = dataset["train"]
    val = dataset["validation"]
    test = dataset["test"]

    example = train[0]
    conversation = example["conversations"]

    all_pairs = []

    def format_conversation(_example):
        turns = [m for m in _example["conversations"] if m["from"] != "system"]

        text = BOS_TOKEN
        for msg in turns:
            text += msg["value"] + EOS_TOKEN

        return text

    def learn_vocab(training):
        text = BOS_TOKEN
        for ex in training:
            conv = ex["conversations"]
            turns = [m for m in conv if m["from"] != "system"]
            for msg in turns:
                text += msg["value"] + EOS_TOKEN

    convo = format_conversation(example)

    llm = LLM(50000, 256, "weights.npz")

    ids = llm.tokenizer.tokenize(convo)
    print(ids)

    for i in range(100):
        print(llm.tokenizer.decode(ids[0:i]))

def test_llm():
    llm = LLM(50000, 256, "weights.npz")
    llm.load_model()
    sentence = "<BOS><USER> hello how are you?<EOS><BOT>"

    for i in range(10):
        pred = llm.forward_pass(sentence)
        token_id = LLM.sample_token(pred)
        token = llm.decode_prediction(token_id)
        sentence += token

    print(sentence)

s = ["<BOS><USER> The quiet street echoed with the sound of distant footsteps at midnight.<EOS><BOT> ",
"<BOS><USER> A single cloud drifted lazily across the bright blue sky.<EOS><BOT> ",
"<BOS><USER> She found an old key hidden inside the pages of a forgotten book.<EOS><BOT> ",
"<BOS><USER> The computer hummed softly as it processed the long sequence of data.<EOS><BOT> ",
"<BOS><USER> Leaves rustled in the wind, creating a rhythm like nature's own music.<EOS><BOT> ",
"<BOS><USER> He paused for a moment, then decided to take the longer path home.<EOS><BOT> ",
"<BOS><USER> The coffee had gone cold, but the conversation was still warm.<EOS><BOT> "]

t = Tokenizer()

for sentence in s:
    print(t.decode(t.encode(sentence)))
