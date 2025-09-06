import spacy
import numpy as np
from tqdm import tqdm
from typing import Tuple


class Preprocessing:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self.stopwords = self.nlp.Defaults.stop_words

    def load_file(self, path: str):
        parsed_data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                split_text = line.split(" ", 1)
                new_item = {"label": split_text[0], "text": split_text[1]}
                parsed_data.append(new_item)
        return parsed_data

    def convert_labels(self, labels: list):
        label_maps = {"__label__1": 0, "__label__2": 1}
        return [label_maps[label] for label in labels]

    def preprocess_text(self, texts: str):
        docs = self.nlp.pipe(texts, batch_size=1000, n_process=4)
        for doc in docs:
            tokens = [
                t.lemma_.lower()
                for t in doc
                if not t.is_punct and t.text.lower() not in self.stopwords
            ]
            yield " ".join(tokens)

    def preprocessing(self, filepath) -> Tuple[list, np.array]:
        print("Starting preprocessing....")
        data_set = self.load_file(filepath)
        print("Processing text....")
        processed_text = list(
            self.preprocess_text(
                item["text"] for item in tqdm(data_set, desc="Processing text data.")
            )
        )
        processed_labels = np.array(
            [
                self.convert_labels(
                    [item["label"] for item in tqdm(data_set, desc="Processing labels")]
                )
            ]
        ).flatten()
        return processed_text, processed_labels
