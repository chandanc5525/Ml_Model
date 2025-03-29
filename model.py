# Import Data Manipulation Libraries
import pandas as pd
import numpy as np

# Import Dataset Using Pandas Function

url = 'https://raw.githubusercontent.com/chandanc5525/Ml_Model/refs/heads/main/Concrete_Data.csv'

df = pd.read_csv(url)

# Split the Dataset into X and y

X = df.drop(columns = 'Concrete compressive strength(MPa, megapascals) ',axis = 1)

y = df['Concrete compressive strength(MPa, megapascals) ']

from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split (X,y,test_size = 0.30,random_state= 42)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)  # Seen Data

X_test = scaler.transform(X_test)        # Unseen Data

# Model Building

from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor()

RF.fit(X_train,y_train)

y_pred_RF = RF.predict(X_test)


from sklearn.metrics import r2_score

r2_score_RF = r2_score(y_test,y_pred_RF)

print(f'The Model R2 Score is {r2_score_RF*100} % ')