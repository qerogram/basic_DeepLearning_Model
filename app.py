import tensorflow as tf
import pandas as pd

class DeepLearning :
  def __init__(self, path, n) :
    self.data = pd.read_csv(path)
    self.독립 = self.data[ [_ for _ in self.data.columns[:n] ] ]    
    self.종속 = self.data[ [_ for _ in self.data.columns[n:] ] ]
    
  
  def compileModel(self) :
    X = tf.keras.layers.Input(shape=[len(self.독립.columns)])
    Y = tf.keras.layers.Dense(len(self.종속.columns))(X)

    self.model = tf.keras.models.Model(X, Y)
    self.model.compile(loss='mse')
    
  
  def learningData(self, epochs=10, verbose=True) :
    self.model.fit(self.독립, self.종속, epochs=epochs, verbose=verbose)
  
  def predictData(self, data) :
    print(self.model.predict(data))


if __name__ == "__main__" :
	d = DeepLearning("https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv", 13)
	d.compileModel()
	d.learningData(epochs=1000, verbose=False)
  d.predictData(d.독립[5:10])
  print(d.종속[5:10])