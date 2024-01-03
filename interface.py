import streamlit as st
import torch
from transformers import BertTokenizerFast

# Loading tokenizer via HuggingFace Transformers
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Loading weights of the model
model = torch.load("c2_new_model_weights.pt")

MAX_LENGHT = 15

# User Interface Components
st.title("FAKE NEWS DETECTOR")
st.write(" A simple fake news detection app utlilizing the capabilities of BERT model via Transformers")
st.subheader("Input the News content below")
sentence = list(st.text_area("Enter your news content here", "Some news", height = 200))

if st.button("Evaluate"):
    tokens_unseen = tokenizer.batch_encode_plus(
    sentence,
    max_length = MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True
)

    unseen_seq = torch.tensor(tokens_unseen['input_ids'])
    unseen_mask = torch.tensor(tokens_unseen['attention_mask'])
    
    with torch.no_grad():
      preds = model(unseen_seq, unseen_mask)
      preds = preds.detach().cpu().numpy()

    preds = np.argmax(preds, axis = 1)
    preds
    st.write(pred)

    