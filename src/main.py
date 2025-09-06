from preprocessing import Preprocessing
from sentiment_analysis import SentimentAnalysis

TRAINING_DATA_PATH = "data/train.ft.txt"
TEST_DATA_PATH = "data/test.ft.txt"


def train_model(training_filepath: str):
    analyzer = SentimentAnalysis()
    preprocessor = Preprocessing()

    # preprocess text data
    x_train, y_train = preprocessor.preprocessing(filepath=training_filepath)

    # fit model
    analyzer.fit_model(x_train, y_train)

    # store model
    analyzer.write_model()


def predict(filepath: str):
    analyzer = SentimentAnalysis()
    preprocessor = Preprocessing()

    # preprocess text data
    corpus, labels = preprocessor.preprocessing(filepath=filepath)

    # predict
    prediction = analyzer.predict(corpus)

    # Show Evaluation metrics
    analyzer.evaluate(prediction, labels)


if __name__ == "__main__":
    print("Start model training....")
    train_model(TRAINING_DATA_PATH)
    print("Finished model training.")
    print("Start testing.")
    predict(TEST_DATA_PATH)
    print("All processes finished.")
