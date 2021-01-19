import tensorflow as tf
import pandas as pd

class DeepLearning :
  REGRESSION_MODE = 1
  CLASSIFICATION_MODE = 2
  # mode = 1 : regression
  # mode = 2 : classification
  def __init__(self, path, n, mode) :
    self.mode = mode
    self.data = pd.read_csv(path)

    # one hot encoding
    if mode == self.CLASSIFICATION_MODE : self.data = pd.get_dummies(self.data)

    self.독립 = self.data[ [_ for _ in self.data.columns[:n] ] ]
    self.종속 = self.data[ [_ for _ in self.data.columns[n:] ] ]
    
  
  def compileModel(self) :
    X = tf.keras.layers.Input(shape=[len(self.독립.columns)])

    ## add Hidden Layer
    H = tf.keras.layers.Dense(10, activation='swish')(X)

    if self.mode == self.REGRESSION_MODE :
      Y = tf.keras.layers.Dense(len(self.종속.columns))(H)
    else :
      Y = tf.keras.layers.Dense(len(self.종속.columns), activation='softmax')(H)

    self.model = tf.keras.models.Model(X, Y)
    self.model.compile(loss='mse' if self.mode == self.REGRESSION_MODE else 'categorical_crossentropy',
                       metrics='accuracy')
    
  
  def learningData(self, epochs=10, verbose=True) :
    self.model.fit(self.독립, self.종속, epochs=epochs, verbose=verbose)
  
  def predictData(self, data) :
    print(self.model.predict(data))

if __name__ == "__main__" :
  d = DeepLearning('https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv', 4, DeepLearning.CLASSIFICATION_MODE)
	d.compileModel()
	d.learningData(epochs=1000, verbose=False)
  d.predictData(d.독립[-5:])
  print(d.종속[-5:])