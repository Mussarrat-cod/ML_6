import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


hours_stud=np.array([2,4,6,8,10,12,14,16,18,20])
exam_sc=np.array([40,50,60,70,80,90,100,110,120,130])

X_train,X_test,Y_train,Y_test= train_test_split(hours_stud.reshape(-1,1),exam_sc, test_size= 0.2, random_state=42)

model= LinearRegression();
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)

mse=mean_squared_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)
print(f'MSE:{mse:.2f}')
print("r2",r2)


