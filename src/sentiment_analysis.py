import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class SentimentAnalysis:
    LOGISTIC_MODEL_PATH = "data/sentiment_model.pkl"
    VECTORIZER_PATH = "data/vectorizer.pkl"

    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = LogisticRegression()

    def load_model(self):
        with open(self.LOGISTIC_MODEL_PATH, "rb") as f:  # pylint: disable=w1514
            self.model = pickle.load(f)

        with open(self.VECTORIZER_PATH, "rb") as f:  # pylint: disable=w1514
            self.vectorizer = pickle.load(f)

    def write_model(self):
        with open(self.LOGISTIC_MODEL_PATH, "wb") as f:  # pylint: disable=w1514
            pickle.dump(self.model, f)

        with open(self.VECTORIZER_PATH, "wb") as f:  # pylint: disable=w1514
            pickle.dump(self.vectorizer, f)

    def fit_model(self, training_vector, y_train: list):
        self.vectorizer.fit(training_vector)
        x_train = self.vectorizer.transform(training_vector)
        self.model.fit(X=x_train, y=y_train)

    def predict(self, test_vector):
        self.load_model()
        x_test = self.vectorizer.transform(test_vector)
        return self.model.predict(x_test)

    def evaluate(self, x_pred, y_pred):
        score = accuracy_score(x_pred, y_pred)
        print("Accuracy Score: ", score)

        report = classification_report(x_pred, y_pred)
        print(report)
