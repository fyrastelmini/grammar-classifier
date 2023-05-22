import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


class Classifier(tf.keras.Model):
    def __init__(self, input_sz, embedding_dim, lstm_units, filters, kernel):
        super(Classifier, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_sz, embedding_dim)
        self.conv1d = tf.keras.layers.Conv1D(filters, kernel, activation='relu')
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.lstm = tf.keras.layers.LSTM(lstm_units)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.embedding(x)
        x = self.conv1d(x)
        x = self.dropout(x)
        x = self.lstm(x)
        output = self.dense(x)
        return output
        
def get_ml_model(name):
	  if name == "Random Forest":
	    return RandomForestClassifier(max_depth=8, random_state=0)
	  elif name == "Naive Bayes Classifier":
	    return GaussianNB()
	  elif name == "Logitic Regression Classifier":
	    return LogisticRegression()
	  elif name == "Support Vector Machine":
	    return SVC()
	  elif name == "Gradient Boost":
	    return GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=24, random_state=0)
	  elif name == "Multilayer Perceptron":
	    return MLPClassifier()
	  else:
	    return 'invalid name: '+name

