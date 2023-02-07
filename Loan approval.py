import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB



#loading the dataset into a pandas dataframe
path="C:/Users/Admin/Desktop/Python_ML_class/ML/loan predector project/Loan Status Prediction.csv"
loan_data = pd.read_csv(path, encoding="ISO-8859-1")
# first five row of the dataset
loan_data.head()

# Print the last five row of the dataset
loan_data.tail()

# checking the dataset shape
loan_data.shape

# cheching the datatype
loan_data.info()

# statisticaly measure
loan_data.describe()


# check the number of missing value in the each columns
loan_data.isna().sum()

# Handeling the missing values---

loan_data.Gender.fillna(value=loan_data.Gender.mode()[0], axis=0, inplace =True )
loan_data.Married.fillna(value=loan_data.Married.mode()[0], axis=0, inplace =True )
loan_data.Dependents.fillna(value=loan_data.Dependents.mode()[0], axis = 0 , inplace = True)
loan_data.Self_Employed.fillna(value=loan_data.Self_Employed.mode()[0], axis = 0 ,inplace =True)
loan_data.LoanAmount.fillna(value = loan_data.LoanAmount.mean(), axis =0 , inplace = True)
loan_data.Loan_Amount_Term.fillna(value = loan_data.Loan_Amount_Term.median(), axis =0 , inplace = True)
loan_data.Credit_History.fillna(value =loan_data.Credit_History.mode()[0], axis =0, inplace=True)


# check the missing value 
loan_data.isnull().sum()

# Dependent column values
loan_data.Dependents.value_counts()


# Dependents columns replaceing value 3+ to 4

loan_data.replace(to_replace="3+" , value="4", inplace = True)
# Dependent column values
loan_data['Dependents'].value_counts()



# drop the loan_id columns
loan_data.drop(columns = "Loan_ID", axis = 1, inplace = True)

#visualisation the data

fig, ax = plt.subplots(2,4, figsize=(16,10))

sns.countplot(x='Loan_Status', data = loan_data, ax=ax[0][0])
sns.countplot(x='Gender', data=loan_data, ax=ax[0][1])
sns.countplot(x='Married', data=loan_data, ax=ax[0][2])
sns.countplot(x='Education', data=loan_data, ax=ax[0][3])
sns.countplot(x='Dependents', data=loan_data, ax=ax[1][0])
sns.countplot(x='Self_Employed', data=loan_data, ax=ax[1][1])
sns.countplot(x='Property_Area', data=loan_data, ax=ax[1][2])
sns.countplot(x='Credit_History', data=loan_data, ax=ax[1][3])


# Education and Loan Status
sns.countplot(x="Education", hue = "Loan_Status", data = loan_data, palette="RdBu")



# Gender and Loan status
sns.countplot(x = "Gender" , hue = "Loan_Status", data= loan_data, palette= "rocket")

# Marital status and Loan Status
sns.countplot(x = "Married" , hue = "Loan_Status", data= loan_data, palette= "CMRmap_r")

# correlation numerical colmns
sns.heatmap(loan_data.corr(), data = loan_data, annot= True , cmap='inferno')

# Distribution numerical variable using the Histogram 

sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

sns.histplot(data=loan_data, x="ApplicantIncome", kde=True, ax=axs[0, 0], color='green')
sns.histplot(data=loan_data, x="CoapplicantIncome", kde=True, ax=axs[0, 1], color='skyblue')
sns.histplot(data=loan_data, x="LoanAmount", kde=True, ax=axs[1, 0], color='orange');
sns.histplot(data=loan_data, x="Loan_Amount_Term", kde=True, ax=axs[1, 1], color='purple');

#Box plot using to show outliers

sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

sns.boxplot(data=loan_data, y="ApplicantIncome", ax=axs[0, 0], color='green')
sns.boxplot(data=loan_data, y="CoapplicantIncome", ax=axs[0, 1], color='skyblue')
sns.boxplot(data=loan_data, y="LoanAmount", ax=axs[1, 0], color='orange');

# Squre root transformation

loan_data.ApplicantIncome = np.sqrt(loan_data.ApplicantIncome)
loan_data.CoapplicantIncome = np.sqrt(loan_data.CoapplicantIncome)
loan_data.LoanAmount = np.sqrt(loan_data.LoanAmount)
loan_data.Loan_Amount_Term = np .sqrt(loan_data.Loan_Amount_Term)
sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

sns.histplot(data=loan_data, x="ApplicantIncome", kde=True, ax=axs[0, 0], color='green')
sns.histplot(data=loan_data, x="CoapplicantIncome", kde=True, ax=axs[0, 1], color='skyblue')
sns.histplot(data=loan_data, x="LoanAmount", kde=True, ax=axs[1, 0], color='orange');
sns.histplot(data=loan_data, x="Loan_Amount_Term", kde=True, ax=axs[1, 1], color='purple');


lab_end = LabelEncoder()
columns =["Gender","Married","Education","Self_Employed", "Property_Area","Loan_Status"] 

loan_data[columns] = loan_data[columns].apply(lab_end.fit_transform)
loan_data.head()


# seprating the data in x and y
x = loan_data.drop(columns = "Loan_Status", axis = 1)
y = loan_data["Loan_Status"]

#Train the dataset
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=2 , stratify=y)

print(x.shape,x_train.shape,x_test.shape)

log_model = LogisticRegression(max_iter=150, solver="liblinear") #'liblinear' library, 'newton-cg', 'sag', 'saga' and 'lbfgs'
log_model.fit(x_train,y_train)

log_prediction = log_model.predict(x_test)

print(classification_report(log_prediction, y_test))
print( confusion_matrix(log_prediction,y_test))

log_acc = accuracy_score(log_prediction,y_test)
print("Logistic Regression accuracy_score: {:.2f}% ". format(log_acc*100))






svm_model = SVC(kernel="rbf",gamma ='auto', C = 6) #'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' # auto,scale
svm_model.fit(x_train, y_train)

svm_prediction = svm_model.predict(x_test)

print(classification_report(svm_prediction,y_test))
print(confusion_matrix(svm_prediction,y_test))

svm_acc = accuracy_score(svm_prediction,y_test)
print("SVM accuracy_score : {:.2f}%".format(svm_acc*100))






dt_model = DecisionTreeClassifier(criterion='gini',splitter = "random")
dt_model.fit(x_train,y_train)

dt_prediction = dt_model.predict(x_test)

print(classification_report(dt_prediction, y_test))
print( confusion_matrix(dt_prediction,y_test))

dt_acc = accuracy_score(dt_prediction,y_test)
print("Decision_tree accuracy_score: {:.2f}% ". format(dt_acc*100))




rf_model = RandomForestClassifier(criterion='entropy', n_estimators=120)
rf_model.fit(x_train,y_train)

rf_prediction = rf_model.predict(x_test)

print(classification_report(rf_prediction, y_test))
print( confusion_matrix(rf_prediction,y_test))

rf_acc = accuracy_score(rf_prediction,y_test)
print("Random_forest accuracy_score: {:.2f}% ". format(rf_acc*100))




knn_model = KNeighborsClassifier(n_neighbors=13)
knn_model.fit(x_train,y_train)

knn_prediction = knn_model.predict(x_test)

print(classification_report(knn_prediction, y_test))
print( confusion_matrix(knn_prediction,y_test))

knn_acc = accuracy_score(knn_prediction,y_test)
print("KNN accuracy_score: {:.2f}% ". format(knn_acc*100))



NBclassifier1 = CategoricalNB()
NBclassifier1.fit(x_train, y_train)

NBclassifier1_prediction = NBclassifier1.predict(x_test)

print(classification_report(NBclassifier1_prediction, y_test))
print( confusion_matrix(NBclassifier1_prediction,y_test))

NBclassifier1_acc = accuracy_score(NBclassifier1_prediction,y_test)
print("Categorical NB accuracy_score: {:.2f}% ". format(NBclassifier1_acc*100))





compare_model = pd.DataFrame({'Model': ['Logistic Regression',"Support Vector Machine", "Decision Tree", 
                             "Random Forest", "K-Nearest Neighbour", " Categorical NB"],
                  'Accuracy_Score': [log_acc*100,svm_acc*100,dt_acc*100,rf_acc*100,knn_acc*100,
                                     NBclassifier1_acc*100]})


compare_model.sort_values(by='Accuracy_Score', ascending=False)








