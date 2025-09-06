import spacy
import numpy as np
from tqdm import tqdm
from typing import Tuple


class Preprocessing:
    """
    A class to handle text preprocessing for sentiment analysis or NLP tasks.

    Attributes:
        nlp (spacy.lang): spaCy NLP pipeline for tokenization and lemmatization.
        stopwords (set): Set of stopwords to remove during preprocessing.
    """

    def __init__(self):
        """
        Initialize the Preprocessing class.

        Loads the small English spaCy model ('en_core_web_sm') with
        parser and NER disabled for faster processing.
        """
        # Load spaCy model without parser and NER for speed
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        # Store stopwords for filtering
        self.stopwords = self.nlp.Defaults.stop_words

    def load_file(self, path: str) -> list[dict]:
        """
        Load a text file and parse each line into a dictionary containing label and text.

        Args:
            path (str): Path to the text file. Each line should start with a label followed by text.

        Returns:
            list[dict]: List of dictionaries with 'label' and 'text' keys.
        """
        parsed_data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                # Split line into label and text
                split_text = line.split(" ", 1)
                new_item = {"label": split_text[0], "text": split_text[1]}
                parsed_data.append(new_item)
        return parsed_data

    def convert_labels(self, labels: list[str]) -> list[int]:
        """
        Convert string labels into numerical labels.

        Args:
            labels (list[str]): List of string labels (e.g., '__label__1').

        Returns:
            list[int]: List of corresponding numerical labels (e.g., 0 or 1).
        """
        # Mapping from string labels to numerical labels
        label_maps = {"__label__1": 0, "__label__2": 1}
        return [label_maps[label] for label in labels]

    def preprocess_text(self, texts: str):
        """
        Generator to preprocess a list of texts.

        Steps:
        - Tokenize using spaCy.
        - Lemmatize tokens.
        - Remove punctuation and stopwords.
        - Convert to lowercase.

        Args:
            texts (Iterable[str]): List or generator of raw text strings.

        Yields:
            str: Preprocessed text as a single string of tokens.
        """
        # Process texts in batches for efficiency
        docs = self.nlp.pipe(texts, batch_size=1000, n_process=4)
        for doc in docs:
            # Lemmatize, lowercase, and remove stopwords/punctuation
            tokens = [
                t.lemma_.lower()
                for t in doc
                if not t.is_punct and t.text.lower() not in self.stopwords
            ]
            yield " ".join(tokens)

    def preprocessing(self, filepath: str) -> Tuple[list[str], np.ndarray]:
        """
        Complete preprocessing pipeline: load file, process text, and convert labels.

        Args:
            filepath (str): Path to the text file to preprocess.

        Returns:
            Tuple[list[str], np.ndarray]: Tuple containing:
                - processed_text: List of preprocessed text strings
                - processed_labels: Numpy array of numerical labels
        """
        print("Starting preprocessing....")
        # Load and parse the dataset
        data_set = self.load_file(filepath)

        print("Processing text....")
        # Preprocess all text data
        processed_text = list(
            self.preprocess_text(
                item["text"] for item in tqdm(data_set, desc="Processing text data.")
            )
        )

        # Convert all labels to numerical format
        processed_labels = np.array(
            [
                self.convert_labels(
                    [item["label"] for item in tqdm(data_set, desc="Processing labels")]
                )
            ]
        ).flatten()

        return processed_text, processed_labels
