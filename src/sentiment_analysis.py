import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class SentimentAnalysis:
    """
    A class to perform sentiment analysis using CountVectorizer and Logistic Regression.

    Attributes:
        LOGISTIC_MODEL_PATH (str): Path to save/load the trained logistic regression model.
        VECTORIZER_PATH (str): Path to save/load the trained CountVectorizer.
        vectorizer (CountVectorizer): Vectorizer to convert text into numerical features.
        model (LogisticRegression): Logistic regression classifier for sentiment prediction.
    """

    LOGISTIC_MODEL_PATH = "data/sentiment_model.pkl"
    VECTORIZER_PATH = "data/vectorizer.pkl"

    def __init__(self, load_existing_model: bool = True):
        """Initialize the vectorizer and the logistic regression model."""
        self.vectorizer = CountVectorizer()
        self.model = LogisticRegression()

        if load_existing_model:
            self.load_model()

    def load_model(self):
        """
        Load the trained model and vectorizer from disk.

        This allows the class to use pre-trained models for prediction
        without retraining.
        """
        with open(self.LOGISTIC_MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

        with open(self.VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)

    def write_model(self):
        """
        Save the trained model and vectorizer to disk.

        This allows the model to be reused later without retraining.
        """
        with open(self.LOGISTIC_MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)

        with open(self.VECTORIZER_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def fit_model(self, training_texts: list[str], y_train: list[int]):
        """
        Train the CountVectorizer and Logistic Regression model on the given data.

        Args:
            training_texts (list[str]): List of raw text documents for training.
            y_train (list[int]): Corresponding sentiment labels for training data.
        """
        # Fit the vectorizer and transform the training texts into numerical features
        x_train = self.vectorizer.fit_transform(training_texts)
        # Train the logistic regression model
        self.model.fit(X=x_train, y=y_train)
        # Save the trained model and vectorizer for later use
        self.write_model()

    def predict(self, test_texts: list[str]) -> list[int]:
        """
        Predict sentiment labels for a list of texts using the trained model.

        Args:
            test_texts (list[str]): List of raw text documents to classify.

        Returns:
            list[int]: Predicted sentiment labels.
        """
        # Transform the test texts into feature vectors
        x_test = self.vectorizer.transform(test_texts)
        # Return predicted labels
        return self.model.predict(x_test)

    def evaluate(self, y_true: list[int], y_pred: list[int]):
        """
        Evaluate the performance of the model using accuracy and classification metrics.

        Args:
            y_true (list[int]): True sentiment labels.
            y_pred (list[int]): Predicted sentiment labels.
        """
        # Calculate accuracy
        score = accuracy_score(y_true, y_pred)
        print("Accuracy Score:", score)

        # Print detailed classification metrics
        report = classification_report(y_true, y_pred)
        print(report)
