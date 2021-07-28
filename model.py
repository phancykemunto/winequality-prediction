# Import Dependencies
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import data from csv file
data=pd.read_csv('C:/PYDATAFILES/winequality.csv')


# Split data into output and input
x=data.iloc[:,:-1]
y=data.iloc[:,-1:]

# Split data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,np.array(y).flatten(),test_size=0.1,random_state=1)
model =RandomForestClassifier()
model.fit(x_train,y_train)

predic=model.predict(x_test)
accuracy_score(y_test,predic)

# Save model using pickle
pickle.dump(model, open('model.pkl','wb'))