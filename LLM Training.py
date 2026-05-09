import os
from LLM import LLM, Tokenizer
from datasets import load_dataset

VOCAB_SIZE    = 50000
EMBEDDING_DIM = 256
WEIGHTS_PATH  = "weights.npz"
VOCAB_PATH    = "tokens_weights.txt"
MAX_SENTENCES = 5
EPOCHS        = 10


def format_conversation(example):
    turns = [m for m in example["conversations"] if m["from"] != "system"]
    text  = "<BOS>"
    for msg in turns:
        role  = "USER" if msg["from"].upper() == "HUMAN" else "BOT"
        text += f"<{role}>{msg['value']}<EOS>"
    return text


def learn_vocab():
    """Run once to build the BPE vocabulary from the full training corpus."""
    print("Loading dataset for vocab...")
    dataset = load_dataset("agentlans/li2017dailydialog")
    train   = dataset["train"]

    text = ""
    for ex in train:
        for msg in ex["conversations"]:
            if msg["from"] != "system":
                text += f" {msg['value']}"

    print(f"Building vocabulary (target size {VOCAB_SIZE})...")
    llm = LLM(VOCAB_SIZE, EMBEDDING_DIM, WEIGHTS_PATH)
    llm.create_vocabulary(text)
    print("Vocabulary saved.")


def get_sentences(max_sentences=None):
    """Return a list of token-id sequences for every conversation in train."""
    print("Loading dataset for training...")
    dataset   = load_dataset("agentlans/li2017dailydialog")
    train     = dataset["train"]
    tokenizer = Tokenizer()

    print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")

    sentences = []
    for ex in train:
        raw = format_conversation(ex)
        ids = tokenizer.encode(raw)
        if len(ids) < 2:
            continue
        sentences.append(ids)

        if len(sentences) % 500 == 0:
            print(f"  tokenized {len(sentences)} conversations...")

        if max_sentences and len(sentences) >= max_sentences:
            break

    print(f"Total conversations loaded: {len(sentences)}")
    return sentences


def train_llm():
    if not os.path.isfile(VOCAB_PATH):
        print("No vocabulary file found — building BPE vocab first.")
        learn_vocab()
    else:
        print(f"Vocabulary file found ({VOCAB_PATH}), skipping BPE step.")

    tokenizer = Tokenizer()
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    llm = LLM(VOCAB_SIZE, EMBEDDING_DIM, WEIGHTS_PATH)
    llm.load_model()
    print("LLM initialised.")

    sentences = get_sentences()

    print("Starting training...")
    llm.train(sentences, epochs=EPOCHS)

def test_llm():
    llm = LLM(VOCAB_SIZE, EMBEDDING_DIM, WEIGHTS_PATH)
    llm.load_model()
    print("Model loaded for testing.")

    sentences = get_sentences(max_sentences=5)
    llm.test(sentences)


if __name__ == "__main__":
    llm = LLM(VOCAB_SIZE, EMBEDDING_DIM, WEIGHTS_PATH)
    llm.load_model()
    sentence = "<BOS><USER>Is it good?<EOS>"

    pred = ""

    while pred != "<EOS>":
        pred = llm.decode_prediction(llm.forward_pass(sentence))
        if pred == "<UNK>":
            pred = ""
        sentence += pred

    print(sentence)
