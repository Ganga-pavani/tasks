import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_data.head()
train_data.info()
train_data = train_data.drop(columns=['keyword', 'location'])
train_data.head()
sns.countplot(x='target', data=train_data,)
plt.title('Target Distribution');

print(train_data['target'].value_counts())
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
english_stopwords = stopwords.words('english')
", ".join(english_stopwords)
stemmer = SnowballStemmer(language = 'english')
def tokenize(text):
  return [stemmer.stem(token) for token in word_tokenize(text) if token.isalpha()]
  from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['text'])
y_train = train_data['target']
X_test = vectorizer.transform(test_data['text'])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred_val = lr_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
print('Validation accuracy:', val_accuracy)
y_pred_test = lr_model.predict(X_test)
submission = pd.DataFrame({'id': test_data['id'], 'target': y_pred_test})
submission.to_csv('submission.csv', index=False)
