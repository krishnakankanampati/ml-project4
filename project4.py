import pandas as pd
df=pd.read_csv('Data.csv')
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
X=X.values
Y=Y.values
from sklearn.impute import SimpleImputer
import numpy as np

si=SimpleImputer(missing_values=np.nan,strategy='mean')
X[:,1:3]=si.fit_transform(X[:,1:3])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X=ct.fit_transform(X)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y=le.fit_transform(Y)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X=ss.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
X_train.shape
X_test.shape
 
