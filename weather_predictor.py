from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

import pandas as pd

data=pd.DataFrame({
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
    'RainTomorrow': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
})

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in data.columns:
    data[col]=le.fit_transform(data[col])

X=data.drop('RainTomorrow',axis=1)
y=data['RainTomorrow']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

model=GaussianNB()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,y_pred)*100, "%")
print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))
