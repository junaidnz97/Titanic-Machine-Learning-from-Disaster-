import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")


'''print(df_train.isnull().sum())
print(set(df_train["Age"]))
print(df_train.shape)
print(df_train["Age"].mode().iloc[0])
print(df_train["Age"].mean())
print(df_train["Age"].value_counts())'''


df_train["Age"]=df_train["Age"].fillna(df_train["Age"].mean())
df_test["Age"]=df_test["Age"].fillna(df_train["Age"].mean())




'''for i in df_train:
	print(i,df_train[i].dtypes)

print(df_train["Embarked"].value_counts())'''

df_train["Embarked"]=df_train["Embarked"].fillna(df_train["Embarked"].mode().iloc[0])
df_test["Embarked"]=df_test["Embarked"].fillna(df_train["Embarked"].mode().iloc[0])

df_train["Cabin"]=df_train["Cabin"].fillna(df_train["Cabin"].mode().iloc[0])
df_test["Cabin"]=df_test["Cabin"].fillna(df_train["Cabin"].mode().iloc[0])

def split_name(x):
	return x.split(",")[1].split(".")[0].split(" ")[1]
	

df_train["Name"]=df_train["Name"].apply(split_name)
df_test["Name"]=df_test["Name"].apply(split_name)


'''object_type=[]

for i in df_train:
	if(df_train[i].dtypes=="object"):
		object_type.append(i)

#print(object_type)


#print((set(df_train["Sex"])))
#print((set(df_train["Sex"])))
'''

def split_cabin_level(x):
	return x[0]

df_train["Cabin_level"]=df_train["Cabin"].apply(split_cabin_level)
df_test["Cabin_level"]=df_test["Cabin"].apply(split_cabin_level)





object_type=['Name', 'Sex',"Embarked","Cabin_level"]
df_test["Name"].replace(to_replace="Dona",value="Don",inplace=True)
for i in object_type:
	le=preprocessing.LabelEncoder()
	le.fit(df_train[i])
	df_train[i]=le.transform(df_train[i])
	df_test[i]=le.transform(df_test[i])


'''class_fare={"1":0.0,"2":0.0,"3":0.0}
count_fare={"1":0,"2":0,"3":0}
for i in range(0,len(df_train)):
	#print(df_train["Pclass"][i])
	count_fare[str(df_train["Pclass"][i])]=count_fare[str(df_train["Pclass"][i])]+1
	class_fare[str(df_train["Pclass"][i])]=class_fare[str(df_train["Pclass"][i])]+df_train["Fare"][i]
class_fare["1"]=class_fare["1"]/count_fare["1"]
class_fare["2"]=class_fare["2"]/count_fare["2"]
class_fare["3"]=class_fare["3"]/count_fare["3"]
print(class_fare)
#print(df_test["Fare"].isnull())
print(df_test["Fare"][152])
print(df_test["Pclass"][152])'''

#df_test["Fare"][152]=13.675550101832997

df_test["Fare"].fillna(13.675550101832997,inplace=True)

#print(set(df_train["Name"]))
#print(set(df_test["Name"]))

def split_ticket(x):
	
	'''y=re.findall("\d+",x)
	print(x)'''
	#print(x[-1: :-1]," ",x)
	y=x[-1: :-1].split()
	try:
		y=int(y[0])
	except:
		y=-1
	return y


#print(Counter(df_train["Ticket"]).most_common())

df_train["Ticket"]=df_train["Ticket"].apply(split_ticket)
df_test["Ticket"]=df_test["Ticket"].apply(split_ticket)


features=["Pclass","Name","Sex","Age","Embarked","SibSp","Parch","Fare","Ticket"]


#print(df_test.dtypes)



xtrain=np.array(df_train[features])
ytrain=np.array(df_train["Survived"])
#print(xtrain)

xtest=np.array(df_test[features])


#print(df_test.isnull().sum())




X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.4, random_state=0)


scaler = MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)




clf=AdaBoostClassifier(n_estimators=300)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


