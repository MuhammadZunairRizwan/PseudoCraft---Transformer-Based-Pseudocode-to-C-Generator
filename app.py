import streamlit as st
import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import base64
from PIL import Image

# ==========================
# 1. COVER PAGE
# ==========================
image_path = "/content/PseudotoCppCover.jpg"
def get_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
base64_image = get_base64(image_path)
st.set_page_config(page_title="Pseudocode to C++ Converter", layout="centered")
st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    [data-testid="stAppViewContainer"] {{
        color: rgb(0, 0, 0);  !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Pseudocode to C++ Code Generator")
st.write("Convert pseudocode into C++ code using a Transformer model.")

# ==========================
# 2. LOAD MODEL & VOCAB
# ==========================

def tokenize_text(text):
    return list(text)

# Load dataset (for vocab)
df_train = pd.read_csv("spoc-train.csv")
df_train.dropna(inplace=True)
data_pairs = list(zip(df_train[' text'].tolist(), df_train['code'].tolist()))

# Build vocabularies
special_tokens = ["<pad>", "<sos>", "<eos>"]
src_vocab = build_vocab_from_iterator((tokenize_text(pair[0]) for pair in data_pairs), specials=special_tokens)
tgt_vocab = build_vocab_from_iterator((tokenize_text(pair[1]) for pair in data_pairs), specials=special_tokens)

src_vocab.set_default_index(src_vocab["<pad>"])
tgt_vocab.set_default_index(tgt_vocab["<pad>"])
idx_to_tgt = {idx: token for token, idx in tgt_vocab.get_stoi().items()}

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, N, h, dropout, d_ff):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 200, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=h,
            num_encoder_layers=N,
            num_decoder_layers=N,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, src, tgt):
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1]).to(src.device)
        src_emb = self.encoder_embedding(src) + self.positional_encoding[:, :src.shape[1], :]
        tgt_emb = self.decoder_embedding(tgt) + self.positional_encoding[:, :tgt.shape[1], :]
        output = self.transformer(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc_out(output)

# Load model
model = TransformerModel(len(src_vocab), len(tgt_vocab), 256, 4, 8, 0.1, 512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("transformer_model.pth", map_location=device))
model.to(device)
model.eval()

# ==========================
# 3. TRANSLATION FUNCTION
# ==========================

def translate_pseudocode(pseudocode, max_length=200):
    model.eval()
    src_tokens = ["<sos>"] + tokenize_text(pseudocode) + ["<eos>"]
    src_indices = [src_vocab[token] for token in src_tokens] + [src_vocab["<pad>"]] * (max_length - len(src_tokens))
    src_tensor = torch.tensor([src_indices], device=device)
    tgt_indices = [tgt_vocab["<sos>"]]
    tgt_tensor = torch.tensor([tgt_indices], device=device)

    for _ in range(max_length):
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
            next_token = output[:, -1, :].argmax(dim=-1).item()
        tgt_indices.append(next_token)
        if next_token == tgt_vocab["<eos>"]:
            break
        tgt_tensor = torch.tensor([tgt_indices], device=device)

    return "".join(idx_to_tgt[idx] for idx in tgt_indices[1:-1])

# ==========================
# 4. STREAMLIT UI
# ==========================

st.subheader("Convert Pseudocode to C++")
pseudocode_input = st.text_area("Enter Pseudocode:", "in the function gcd(a,b=integers)")
if st.button("Convert"):
    with st.spinner("Generating C++ Code..."):
        generated_code = translate_pseudocode(pseudocode_input)
    st.success("Generated C++ Code:")
    st.code(generated_code, language="cpp")
