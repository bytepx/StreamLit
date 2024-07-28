# %%
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#seaborn
import seaborn as sns
# matplotlib
import matplotlib.pyplot as plt

# %%
data=pd.read_csv("car data.csv")
data.head()

# %%
data.Car_Name.value_counts()

# %%
data.info()


# %%
sns.heatmap(data.isnull(),annot=True)


# %%
sns.heatmap(data.isna(),annot=True)


# %%
print(data.Fuel_Type.value_counts(),"\n")
print(data.Seller_Type.value_counts(),"\n")
print(data.Transmission.value_counts())

# %%
data.Fuel_Type.replace(regex={"Petrol":"0","Diesel":"1","CNG":"2"},inplace=True)
data.Seller_Type.replace(regex={"Dealer":"0","Individual":"1"},inplace=True)
data.Transmission.replace(regex={"Manual":"0","Automatic":"1"},inplace=True)
data[["Fuel_Type","Seller_Type","Transmission"]]=data[["Fuel_Type","Seller_Type","Transmission"]].astype(int)

# %%
sns.pairplot(data,diag_kind="kde", diag_kws=dict(shade=True, bw=.05, vertical=False))
plt.show()

# %%
y=data.Selling_Price
x=data.drop(["Selling_Price","Car_Name"],axis=1)


# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

# %%
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()


# %%
dtr.fit(x_train,y_train)

# %%
y_pred=dtr.predict(x_test)

# %%
from sklearn import metrics
metrics.mean_squared_error(y_test, y_pred)

# %%
import pickle
model=open("car price.pkl","wb")
pickle.dump(dtr,model)
model.close()

# %%


# %%



