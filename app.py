import streamlit as st
import torch
import numpy as np
import pickle
from string import punctuation
from model import SentimentRNN  # optionally move class into model.py

# Set this to False unless you specifically want GPU
train_on_gpu = False

# Load vocab
with open('vocab_to_int.pkl', 'rb') as f:
    vocab_to_int = pickle.load(f)

# Tokenization
def tokenize_article(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in punctuation])
    words = text.split()
    return [[vocab_to_int.get(word, 0) for word in words]]

# Padding
def pad_features(text_ints, seq_length):
    features = np.zeros((len(text_ints), seq_length), dtype=int)
    for i, row in enumerate(text_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = SentimentRNN(
        vocab_size=len(vocab_to_int) + 1 ,
        output_size=1,
        embedding_dim=200,
        hidden_dim=256,
        n_layers=3
    )
    model.load_state_dict(torch.load('checkpoint6.pth', map_location = device))
    model.to(device)
    model.eval()

    return model

device = torch.device('cpu')

model = load_model()

# Streamlit interface
st.title("Information Analyzer")

user_input = st.text_area("Enter your Post or Information:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        test_ints = tokenize_article(user_input)
        seq_length = 200
        padded = pad_features(test_ints, seq_length)
        input_tensor = torch.tensor(padded).long()

        device = torch.device('cpu')
        input_tensor = input_tensor.to(device)


        batch_size = input_tensor.size(0)
        h = model.init_hidden(batch_size, train_on_gpu)

        with torch.no_grad():
            output, _ = model(input_tensor, h)
            pred = int(torch.round(output).item())
            confidence = output.item()

        if pred == 0:
            st.write(f"Prediction: ** Reliable Information or Post ** (Confidence: {confidence:.2f})")

        else:
            st.write(f"Prediction: ** Not Reliable Information or Post ** (Confidence: {confidence:.2f})")
