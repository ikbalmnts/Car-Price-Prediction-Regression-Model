import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error

class CarPrediction:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.scaler = MinMaxScaler()
    
    def Data_Preprocessing(self):
        #Data is edited.
        print(self.data.describe())
        print(self.data.isnull().sum())
        self.data = self.data.drop(["transmission", "model", "fuelType"], axis=1)
        self.data = self.data.sort_values(["price"], ascending=False).iloc[179:]
        self.data = self.data.drop(self.data[self.data["year"] == 2060].index)
    
    def Data_Visualization(self):
        #The price column of the data is visualized.
        sns.displot(self.data["price"], height=6, aspect=2)
        plt.show()
    
    def Model_Training(self):
        #The model is created and necessary operations are performed
        y = self.data["price"].values
        X = self.data.drop(["price"], axis=1).values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model = Sequential()
        model.add(Dense(15, activation="relu"))
        model.add(Dense(15, activation="relu"))
        model.add(Dense(15, activation="relu"))
        model.add(Dense(15, activation="relu"))
        model.add(Dense(15, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=50)
        model.fit(x=X_train_scaled, y=y_train, validation_data=(X_test_scaled, y_test), batch_size=250, epochs=300, verbose=1, callbacks=[early_stopping])
        
        self.model = model
        self.X_test = X_test_scaled
        self.y_test = y_test
    
    def Result_Graph(self):
        #The accuracy of the model is checked by visualization.
        history = pd.DataFrame(self.model.history.history)
        history.plot()
        plt.show()
    
    def Evaluation(self):
        #The margin of deviation is checked
        predict = self.model.predict(self.X_test)
        print("Mean Absolute Error:", mean_absolute_error(self.y_test, predict))
    
    def Predict_Price(self, index):
        #The model is run by sampling.
        car_features = self.data.drop("price", axis=1).iloc[index]
        car_features_scaled = self.scaler.transform(car_features.values.reshape(1, -1))
        print(self.data.iloc[index])
        print("Predicted Price:", self.model.predict(car_features_scaled))

model = CarPrediction("ford.csv")
model.Data_Preprocessing()
model.Data_Visualization()
model.Model_Training()
model.Result_Graph()
model.Evaluation()
model.Predict_Price(200)