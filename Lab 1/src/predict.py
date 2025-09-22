import torch
from .model import BERT_SPO_BIO_Tagger, tokenizer, device, id2tag
import spacy

# Load spaCy English language model for text tokenization and sentence segmentation
nlp = spacy.load("en_core_web_sm")

# Initialize and load the pre-trained BERT model for BIO tagging
model = BERT_SPO_BIO_Tagger().to(device)
model.load_model_weights(torch.load("./model/BERT_BIO_Tagging_model.pth"))

def tokenize_text(text):
    """
    Tokenize input text using spaCy for consistent word-level tokenization.
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: List of token strings
    """
    return [token.text for token in nlp(text)]

@torch.no_grad()  # Disable gradient computation for inference
def predict_bio_tags(text, model, tokenizer, id2tag, device):
    """
    Predict BIO tags for each token in the input text using the BERT model.
    
    This function processes the input text through the BERT model to predict
    whether each token is part of a Subject (B-SUB, I-SUB), Predicate (B-PRED, I-PRED),
    Object (B-OBJ, I-OBJ), or Outside (O) entity.
    
    Args:
        text (str): Input text to process
        model: Pre-trained BERT model
        tokenizer: BERT tokenizer
        id2tag (dict): Mapping from tag IDs to tag names
        device: PyTorch device (CPU/GPU)
        
    Returns:
        list: List of (token, tag) tuples
    """
    # Set model to evaluation mode for inference
    model.eval()
    
    # Tokenize the input text into individual words
    tokens = tokenize_text(text)

    # Tokenize using BERT tokenizer with word-level tokenization
    tokenized_input = tokenizer(tokens,
                                is_split_into_words=True,
                                return_tensors="pt",
                                truncation=True,
                                padding="max_length",
                                max_length=512)

    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(logits, dim=2)

    word_ids = tokenized_input.word_ids(batch_index=0)
    predicted_tags = []

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        tag_id = predictions[0][idx].item()
        tag = id2tag[tag_id]
        if idx == 0 or word_ids[idx] != word_ids[idx - 1]:
            predicted_tags.append((tokens[word_idx], tag))

    return predicted_tags

def extract_and_form_triplets(text, tagged_tokens):
    """
    Extract subject-predicate-object triplets from BIO-tagged tokens.
    
    This function processes the BIO-tagged tokens to identify continuous spans
    of subjects, predicates, and objects, then forms all possible combinations
    of triplets within the same sentence.
    
    Args:
        text (str): Original input text
        tagged_tokens (list): List of (token, tag) tuples from BIO tagging
        
    Returns:
        list: List of (subject, predicate, object) triplets
    """

    # Step 1: Extract continuous spans from BIO-tagged tokens
    # Dictionary to store all identified spans for each entity type
    spans = {'SUB': [], 'PRED': [], 'OBJ': []}
    current_span = []
    current_label = None

    for token, tag in tagged_tokens:
        if tag == 'O':
            if current_span and current_label:
                spans[current_label].append(" ".join(current_span))
            current_span = []
            current_label = None
        elif tag.startswith('B-'):
            if current_span and current_label:
                spans[current_label].append(" ".join(current_span))
            current_label = tag[2:]
            current_span = [token]
        elif tag.startswith('I-') and current_label == tag[2:]:
            current_span.append(token)
        else:
            if current_span and current_label:
                spans[current_label].append(" ".join(current_span))
            current_span = []
            current_label = None

    if current_span and current_label:
        spans[current_label].append(" ".join(current_span))

    # Step 2: Filter out short or lowercase-only spans
    # def filter_spans(spans):
    #     def is_valid(span):
    #         return len(span.split()) > 1 or span[0].isupper()

    #     return {
    #         k: [s for s in v if is_valid(s)] for k, v in spans.items()
    #     }

    # filtered_spans = filter_spans(spans)

    filtered_spans = spans

    # Step 3: Match spans within same sentence
    doc = nlp(text)
    triplets = []

    for sent in doc.sents:
        sent_text = sent.text
        subjs = [s for s in filtered_spans["SUB"] if s in sent_text]
        preds = [p for p in filtered_spans["PRED"] if p in sent_text]
        objs = [o for o in filtered_spans["OBJ"] if o in sent_text]

        for s in subjs:
            for p in preds:
                for o in objs:
                    triplets.append((s, p, o))

    return list(set(triplets))

def predict_triplets(input_str):
    """
    Main function to extract subject-predicate-object triplets from input text.
    
    This is the main entry point that orchestrates the entire triplet extraction process:
    1. Uses BERT model to predict BIO tags for each token
    2. Extracts continuous spans of subjects, predicates, and objects
    3. Forms all possible triplets within the same sentence
    
    Args:
        input_str (str): Input text to process
        
    Returns:
        list: List of (subject, predicate, object) triplets
    """
    tagged = predict_bio_tags(input_str, model, tokenizer, id2tag, device)
    triplets = extract_and_form_triplets(input_str, tagged)
    return triplets
