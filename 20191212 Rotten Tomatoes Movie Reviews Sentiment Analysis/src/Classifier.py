import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

class Base:
    """Base class that houses common utilities for reading in test data
    and calculating model accuracy and F1 scores.
    """
    def __init__(self) -> None:
        pass

    def read_data(self, fname: str, lower_case: bool=False) -> pd.DataFrame:
        "Read in test data into a Pandas DataFrame"
        df = pd.read_csv(fname, sep=',')
        
        df['sentiment'] = df['sentiment'].astype(str).str.replace('__label__', '')
        # Categorical data type for sentiment labels
        df['sentiment'] = df['sentiment'].astype(int).astype('category')
        # Optional lowercase for test data (if model was trained on lowercased text)
        if lower_case:
            df['sentence'] = df['sentence'].str.lower()
        return df

    def accuracy(self, df: pd.DataFrame) -> None:
        "Prediction accuracy (percentage) and F1 score"
        acc = accuracy_score(df['sentiment'], df['pred'])*100
        f1 = f1_score(df['sentiment'], df['pred'], average='macro')*100
        print("Accuracy: {:.3f}\nMacro F1-score: {:.3f}".format(acc, f1))


class TextBlobSentiment(Base):
    """Predict sentiment scores using TextBlob.
    https://textblob.readthedocs.io/en/dev/
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()

    def score(self, text: str) -> float:
        # pip install textblob
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity

    def predict(self, train_file: None, test_file: str, lower_case: bool) -> pd.DataFrame:
        df = self.read_data(test_file, lower_case)
        df['score'] = df['sentence'].apply(self.score)
        # Convert float score to category based on binning
        df['pred'] = pd.cut(df['score'],
                            bins=5,
                            labels=[1, 2, 3, 4, 5])
        df = df.drop('score', axis=1)
        return df


class VaderSentiment(Base):
    """Predict fine-grained sentiment classes using Vader."""
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        return self.vader.polarity_scores(text)['compound']

    def predict(self, train_file: None, test_file: str, lower_case: bool) -> pd.DataFrame:
        "Return DataFrame with a new column of predicted labels"
        df = self.read_data(test_file, lower_case)
        df['score'] = df['sentence'].apply(self.score)
        # Convert float score to category based on binning
        df['pred'] = pd.cut(df['score'], bins=5, labels=[1, 2, 3, 4, 5])
        df = df.drop('score', axis=1)
        return df

class LogisticRegressionSentiment(Base):
    """Predict fine-grained sentiment scores using a sklearn Logistic Regression pipeline."""
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer(stop_words = 'english')),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver='liblinear', multi_class='auto')),
            ]
        )

    def predict(self, train_file: str, test_file: str, lower_case: bool=False) -> pd.DataFrame:
        "Train model using sklearn pipeline"
        train_df = self.read_data(train_file, lower_case)
        learner = self.pipeline.fit(train_df['sentence'], train_df['sentiment'])
        # Predict class labels using the learner and output DataFrame
        test_df = self.read_data(test_file, lower_case)
        test_df['pred'] = learner.predict(test_df['sentence'])
        return test_df

class SVMSentiment(Base):
    """Predict sentiment scores using a linear Support Vector Machine (SVM).
    Uses a sklearn pipeline.
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        # pip install sklearn
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(
                    loss='hinge',
                    penalty='l2',
                    alpha=1e-3,
                    random_state=42,
                    max_iter=100,
                    learning_rate='optimal',
                    tol=None,
                )),
            ]
        )

    def predict(self, train_file: str, test_file: str, lower_case: bool) -> pd.DataFrame:
        "Train model using sklearn pipeline"
        train_df = self.read_data(train_file, lower_case)
        learner = self.pipeline.fit(train_df['sentence'], train_df['sentiment'])
        # Fit the learner to the test data
        test_df = self.read_data(test_file, lower_case)
        test_df['pred'] = learner.predict(test_df['sentence'])
        return test_df