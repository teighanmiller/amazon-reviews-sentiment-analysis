from preprocessing import Preprocessing
from sentiment_analysis import SentimentAnalysis

# File paths for training and testing datasets
TRAINING_DATA_PATH = "data/train.ft.txt"
TEST_DATA_PATH = "data/test.ft.txt"


def train_model(training_filepath: str):
    """
    Train a sentiment analysis model using the given training data.

    Steps:
    1. Preprocess the text and labels using Preprocessing class.
    2. Fit the Logistic Regression model with the processed data.
    3. Save the trained model and vectorizer to disk.

    Args:
        training_filepath (str): Path to the training dataset file.
    """
    # Initialize sentiment analysis and preprocessing objects
    analyzer = SentimentAnalysis()
    preprocessor = Preprocessing()

    # Preprocess text data and convert labels to numerical format
    x_train, y_train = preprocessor.preprocessing(filepath=training_filepath)

    # Train the logistic regression model
    analyzer.fit_model(x_train, y_train)

    # Save the trained model and vectorizer
    analyzer.write_model()


def predict(filepath: str):
    """
    Predict sentiment labels for a given dataset and evaluate model performance.

    Steps:
    1. Preprocess the text and labels using Preprocessing class.
    2. Load the trained sentiment analysis model.
    3. Make predictions on the dataset.
    4. Evaluate predictions using accuracy and classification report.

    Args:
        filepath (str): Path to the dataset file to predict and evaluate.
    """
    # Initialize sentiment analysis and preprocessing objects
    analyzer = SentimentAnalysis(load_existing_model=True)
    preprocessor = Preprocessing()

    # Preprocess text data and labels
    corpus, labels = preprocessor.preprocessing(filepath=filepath)

    # Predict sentiment labels using the trained model
    prediction = analyzer.predict(corpus)

    # Evaluate and print performance metrics
    analyzer.evaluate(labels, prediction)


if __name__ == "__main__":
    # Entry point for the script:
    # 1. Train the model on training data.
    # 2. Test the model on testing data.
    print("Start model training....")
    train_model(TRAINING_DATA_PATH)
    print("Finished model training.")

    print("Start testing.")
    predict(TEST_DATA_PATH)
    print("All processes finished.")
