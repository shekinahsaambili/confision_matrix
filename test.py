from sklearn.metrics import confusion_matrix
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#chargement du dataset
dataset=pd.read_csv('iris.csv')

#x et y
x=dataset[["sepal.length","sepal.width","petal.length", "petal.width"]]
y=dataset["variety"]

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.5,random_state=0)

scaler=StandardScaler()

scaler.fit(x_train)
# standardisation ds variables independantes
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

classify= KNeighborsClassifier(n_neighbors=1)

#entrainement

classify.fit(x_train,y_train)

y_predi= classify.predict(x_test)
df=pd.DataFrame({'y_connu':y_test, 'y_predi':y_predi})
print(df)
#knn.fit(X_train, y_train)

# Prédire les étiquettes sur l'ensemble de test
#y_pred = knn.predict(X_test)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_predi)

print("Matrice de confusion :")
print(conf_matrix)
