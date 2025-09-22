import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

# # Unique BIO tags
tag_values = ['O', 'B-SUB', 'I-SUB', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
tag2id = {tag: i for i, tag in enumerate(tag_values)}
id2tag = {i: tag for tag, i in tag2id.items()}

if torch.backends.mps.is_available():
    device = torch.device("mps")
    # print ("MPS device found.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    # print ("CUDA device found.")
else:
    device = torch.device("cpu")
    # print ("Using CPU.")

class BERT_SPO_BIO_Tagger(nn.Module):

    def __init__(self):
        super(BERT_SPO_BIO_Tagger, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(tag2id))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        return logits

    def load_model_weights(self, torch_load_weights):
        self.load_state_dict(torch_load_weights)
