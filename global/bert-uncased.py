import codecs
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


# Define mean pooling function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def transform(byte_arrays, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Assumes byte_arrays is a single byte array (bytes object) correctly representing
    UTF-16 encoded text, and converts it into sentence embeddings.
    """
    print(byte_arrays)
    # Check and remove the UTF-16 BOM if present
    if byte_arrays.startswith(codecs.BOM_UTF16_LE):
        input_text = byte_arrays.decode('utf-16-le')
    elif byte_arrays.startswith(codecs.BOM_UTF16_BE):
        input_text = byte_arrays.decode('utf-16-be')
    else:
        # Default to little-endian if no BOM is present; adjust as needed
        input_text = byte_arrays.decode('utf-16-le')

    print('Input text:', input_text)

    # Load tokenizer and model from the sentence-transformer or Hugging Face model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize and encode the input text
    encoded_input = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')

    # Compute model output
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform mean pooling to get sentence embeddings
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Optionally, you can normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # Convert to numpy array if needed, for compatibility with existing code
    sentence_embeddings_np = sentence_embeddings.numpy()

    return sentence_embeddings_np
