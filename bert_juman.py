from pathlib import Path

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pyknp import Juman


class JumanTokenizer():
    def __init__(self):
        self.juman = Juman()

    def tokenize(self, text):
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]


class BertWithJumanModel():
    def __init__(self, bert_path, vocab_file_name="vocab.txt"):
        self.juman_tokenizer = JumanTokenizer()
        self.model = BertModel.from_pretrained(bert_path)
        self.bert_tokenizer = BertTokenizer(Path(bert_path) / vocab_file_name,
                                            do_lower_case=False, do_basic_tokenize=False)

    def _preprocess_text(self, text):
        return text.replace(" ", "")  # for Juman

    def get_sentence_embedding(self, text, pooling_layer=-2, pooling_strategy="REDUCE_MEAN"):
        preprocessed_text = self._preprocess_text(text)
        tokens = ["[CLS]"] + self.juman_tokenizer.tokenize(preprocessed_text)
        bert_tokens = self.bert_tokenizer.tokenize(" ".join(tokens[:128]))  # max_seq_len
        ids = self.bert_tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor(ids).reshape(1, -1)

        self.model.eval()
        with torch.no_grad():
            all_encoder_layers, _ = self.model(tokens_tensor)

        embedding = all_encoder_layers[pooling_layer].numpy()[0]
        if pooling_strategy == "REDUCE_MEAN":
            return np.mean(embedding, axis=0)
        elif pooling_strategy == "REDUCE_MAX":
            return np.max(embedding, axis=0)
        elif pooling_strategy == "REDUCE_MEAN_MAX":
            return np.r_[np.max(embedding, axis=0), np.mean(embedding, axis=0)]
        else:
            raise ValueError("specify valid pooling_strategy: {REDUCE_MEAN, REDUCE_MAX, REDUCE_MEAN_MAX}")