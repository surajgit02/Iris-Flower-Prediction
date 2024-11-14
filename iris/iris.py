# load  the iris data in program
import pandas as pd
df=pd.read_csv("iris.csv",names=['sl','sw','pl','pw','class'])
# apply labelencoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['class']=le.fit_transform(df['class'])
# devide the data into  input and output
X=df.drop(columns=['class'])
Y=df['class']
# split the data(70% for training  and 30% testing) for training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=5)
################################
#################################
# KNN   model
from sklearn.neighbors import KNeighborsClassifier
K=KNeighborsClassifier(n_neighbors=5)
# train the model by 70% data
K.fit(X_train,Y_train)
# test the model  by 30% data
Y_pred_knn=K.predict(X_test)
#  find accuracy of KNN  model
from sklearn.metrics import accuracy_score
acc_knn=accuracy_score(Y_test,Y_pred_knn)
acc_knn=round(acc_knn*100,2)
print("accuracy in knn is",acc_knn,"%")
# implement logistic regression
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
L=LogisticRegression()
#train the model
L.fit(X_train,Y_train)
# test the model
Y_pred_lg=L.predict(X_test)
# find accuracy
from sklearn.metrics import accuracy_score
acc_lg=accuracy_score(Y_test,Y_pred_lg)
acc_lg=round(acc_lg*100,2)
print("accuracy in logistic regression is ",acc_lg)
#########################################
# implement Decision tree model
from sklearn.tree import DecisionTreeClassifier
D=DecisionTreeClassifier()
D.fit(X_train,Y_train)
Y_pred_dt=D.predict(X_test)
# find accuracy
from sklearn.metrics import accuracy_score
acc_dt=accuracy_score(Y_test,Y_pred_dt)
acc_dt=round(acc_dt*100,2)
print("accuracy in decision tree is ",acc_dt)
########################################
# implement Naive bayes model
from sklearn.naive_bayes import GaussianNB
N=GaussianNB()
N.fit(X_train,Y_train)
Y_pred_nb=N.predict(X_test)
# find accuracy
from sklearn.metrics import accuracy_score
acc_nb=accuracy_score(Y_test,Y_pred_nb)
acc_nb=round(acc_nb*100,2)
print("accuracy in naive bayes is ",acc_nb)
#############
import pickle
f=open('iris.pkl','wb')
pickle.dump(L,f)
f.close()