import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet

#training set & test set
df = pd.read_csv('concrete.csv',header=0)
#print basic information of dataset
print(df.head(),'\n')
print(df.describe())

#make the boxplot
array = df.values
plt.boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel(("Quartile Ranges"))
plt.savefig('boxplot.png')
plt.show()

#make the heatmap
corMat = pd.DataFrame(df.corr())
print(corMat)
_=sns.heatmap(df.corr(), square=True, cmap='RdYlGn',annot=True)
plt.savefig('heatmap.png')

sns.pairplot(df,size=2.5)
plt.tight_layout()
plt.show()

#split the dataset
X=df.iloc[:,0:8].values
y=df.iloc[:,8].values
y=y.reshape((1030,1))
sc_x=StandardScaler()
sc_y=StandardScaler()
sc_x.fit(X)
sc_y.fit(y)
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_std, y_std,test_size=0.2, random_state=42)
print()

def result_show(reg,X_train,X_test,y_train,y_test):
    y_train_pred = reg.predict(X_train)
    y_train_pred=y_train_pred.reshape((-1,1))
    y_test_pred=reg.predict(X_test)
    y_test_pred=y_test_pred.reshape((-1,1))
    print('Coefficient:',reg.coef_)
    print('Intercept:',reg.intercept_)
    print('R^2 train: %.4f, test: %.4f' % (r2_score(y_train, y_train_pred),
                                           r2_score(y_test, y_test_pred)))
    print('MSE train: %.4f, test: %.4f' % (mean_squared_error(y_train, y_train_pred),
                                           mean_squared_error(y_test, y_test_pred)))
    plt.scatter(y_train_pred,  y_train_pred - y_train,
             c='steelblue', marker='o', edgecolor='white',
             label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-2, xmax=3, color='black', lw=2)
    plt.xlim([-2, 3])
    plt.show()
    
#fit the normal linear regression
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
print('Linear Regression without regularization','\n')
result_show(reg,X_train,X_test,y_train,y_test)
print()

#fit the lasso-linear regression
print('Linear Regression with Lasso','\n')
score=[]
k_range=[0.00001,0.0001,0.001,0.01,0.1,1,10]
for k in k_range:
    lasso= Lasso(alpha=k)
    lasso.fit(X_train,y_train)
    y_test_pred=lasso.predict(X_test)
    score.append(r2_score(y_test,y_test_pred))
best_r2_lasso = max(score)
index_lasso = [i for i,x in enumerate(score) if x==best_r2_lasso]    
print('The best alpha is: ', k_range[index_lasso[0]])
lasso1=Lasso(alpha=k_range[index_lasso[0]])
lasso1.fit(X_train,y_train)
result_show(lasso1,X_train,X_test,y_train,y_test)
print()

#fit the riddge-linear regression
print('Linear Regression with Ridge','\n')
score=[]
for k in k_range:
    ridge = Ridge(alpha=k)
    ridge.fit(X_train,y_train)
    y_test_pred=ridge.predict(X_test)
    score.append(r2_score(y_test,y_test_pred))
best_r2_ridge = max(score)
index_ridge = [i for i,x in enumerate(score) if x==best_r2_ridge]
print('The best alpha is: ',k_range[index_ridge[0]])
ridge1=Ridge(alpha=k_range[index_ridge[0]])
ridge1.fit(X_train,y_train)
result_show(ridge1,X_train,X_test,y_train,y_test)
print()


#fit the elasticnet-linear regression
print('Linear Regression with ElasticNet','\n')
score=[]
r_range = np.arange(0,1,0.01).tolist()
for r in r_range:
    elanet = ElasticNet(alpha=1, l1_ratio=r)
    elanet.fit(X_train,y_train)
    y_test_pred=elanet.predict(X_test)
    score.append(r2_score(y_test,y_test_pred))
best_r2_elanet=max(score)
index_elanet = [i for i,x in enumerate(score) if x==best_r2_elanet]
print('The best ratio is: ', r_range[index_elanet[0]])

elanet1=ElasticNet(alpha=1, l1_ratio=0.01)
elanet1.fit(X_train,y_train)
result_show(elanet1,X_train,X_test,y_train,y_test)

print("My name is Bingjie Han")
print("My NetID is: bingjie5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")