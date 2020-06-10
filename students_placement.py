#Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Import the dataset
dataset = pd.read_csv('Placement_Data.csv')

#Changing columns names
dataset.rename(columns = {'sl_no':'serial','ssc_p':'secondary_edu','ssc_b':'board_edu',
                'hsc_p':'higher_second_edu','hsc_b':'board_edu_2','hsc_s':'second_edu_speci',
                'degree_p':'degree','degree_t':'degree_title','workex':'experience',
                'etest_p':'employability_test','specialisation':'MBA_title',
                'mba_p':'MBA','status':'placement'},inplace = True)

#EDA
dataset.info()

#The data is a little unbalanced
sns.countplot('placement',data = dataset)

sns.countplot('placement',hue = 'gender',data = dataset)
sns.countplot('placement',hue = 'board_edu',data = dataset)
sns.countplot('placement',hue = 'board_edu_2',data = dataset)
sns.countplot('placement',hue = 'second_edu_speci',data = dataset)
sns.countplot('placement',hue = 'degree_title',data = dataset)
sns.countplot('placement',hue = 'experience',data = dataset)

no_exp_degree_not_placed = dataset[(dataset['experience'] == 'No')&(dataset['placement'] == 'Not Placed')]['degree_title']
no_exp_degree_placed = dataset[(dataset['experience'] == 'No')&(dataset['placement'] == 'Placed')]['degree_title']
no_exp_board_not_placed = dataset[(dataset['experience'] == 'No')&(dataset['placement'] == 'Not Placed')]['board_edu']
no_exp_board_placed = dataset[(dataset['experience'] == 'No')&(dataset['placement'] == 'Placed')]['board_edu']

sns.countplot(no_exp_degree_not_placed)
sns.countplot(no_exp_degree_placed)
sns.countplot(no_exp_board_not_placed)
sns.countplot(no_exp_board_placed)

sns.countplot('placement',hue = 'MBA_title',data = dataset)

sns.scatterplot(range(0,len(dataset['secondary_edu'])),'secondary_edu',
                hue = 'placement',data = dataset)
sns.scatterplot(range(0,len(dataset['higher_second_edu'])),'higher_second_edu',
                hue = 'placement',data = dataset)
sns.scatterplot(range(0,len(dataset['degree'])),'degree',
                hue = 'placement',data = dataset)
sns.scatterplot(range(0,len(dataset['employability_test'])),'employability_test',
                hue = 'placement',data = dataset)
sns.scatterplot(range(0,len(dataset['MBA'])),'MBA',
                hue = 'placement',data = dataset)

#Data correlation
sns.heatmap(dataset.corr(),annot = True)
#Even though it has high correlation with other features removing the feature
#Result in, not so bad, a little less accuracy con the models.
dataset.drop(['secondary_edu'],axis = 1,inplace = True)

#Since the salary has nan values is not a good choice to make a predictor on the salary
#Since it will be too bias towards the placed ones, theregore it became a classificacion problem

#Checking nan values and dropping more useless columns
dataset.isnull().sum()
dataset.drop(['serial','salary'],axis = 1,inplace = True)

#Encoding the categorical values
label_encoder = LabelEncoder()
dataset['gender'] = label_encoder.fit_transform(dataset['gender'])
dataset['board_edu'] = label_encoder.fit_transform(dataset['board_edu'])
dataset['board_edu_2'] = label_encoder.fit_transform(dataset['board_edu_2'])
dataset['experience'] = label_encoder.fit_transform(dataset['experience'])
dataset['MBA_title'] = label_encoder.fit_transform(dataset['MBA_title'])
dataset['placement'] = label_encoder.fit_transform(dataset['placement'])

columns_to_transform = ['second_edu_speci','degree_title']
transformer = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),columns_to_transform)],
                                remainder = 'passthrough')
X = transformer.fit_transform(dataset.drop(['placement'],axis = 1))
y = dataset['placement']

#Splitting the dataset
X_train, X_test, y_train,y_test = train_test_split(X, y,test_size = 0.20)

#Scaling the features
scaler = ColumnTransformer(transformers=[('scaler',StandardScaler(),[7,9,11,13,15])],
                           remainder = 'passthrough')
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Evaluation metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

#Implementing different models

#Naive Bayes - Mean score = 0.80% - Report score = 81%
from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()

naive_score = cross_val_score(naive,X_train,y_train,cv = 10)
print(np.mean(naive_score))

naive.fit(X_train, y_train)
y_pred = naive.predict(X_test)
print(classification_report(y_test,y_pred))

#Logistics Regression - Mean score = 86%- Report score = 93%
from sklearn.linear_model import LogisticRegression
log_regressor = LogisticRegression()

param_grid = {'C':[0.1,1,10,100]}
grid_search = GridSearchCV(log_regressor,param_grid,cv = 10)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

log_regressor = LogisticRegression(C = 10)
log_score = cross_val_score(log_regressor,X_train,y_train,cv = 10)
print(np.mean(log_score))

log_regressor.fit(X_train, y_train)
y_pred = log_regressor.predict(X_test)
print(classification_report(y_test, y_pred))

#KNeightbors - Mean score = 84% - Report score = 88%
from sklearn.neighbors import KNeighborsClassifier
kneighbor = KNeighborsClassifier()

param_grid = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
grid_search = GridSearchCV(kneighbor,param_grid, cv = 10)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

kneighbor = KNeighborsClassifier(n_neighbors = 14)
kneighbor_score = cross_val_score(kneighbor,X_train,y_train,cv = 10)
print(np.mean(kneighbor_score))

kneighbor.fit(X_train, y_train)
y_pred = kneighbor.predict(X_test)
print(classification_report(y_test,y_pred))

#Support Vector Classifier - Mean score = 88%- Report Score = 93%
from sklearn.svm import SVC
svc = SVC()

param_grid = {'kernel':['linear','rbf','sigmoid'],
              'C':[0.1,1,10,100],
              'gamma':[0.1,1,10,100]}
grid_search = GridSearchCV(svc,param_grid, cv = 10)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

svc = SVC(C = 1,gamma = 0.1,kernel = 'linear')
svc_score = cross_val_score(svc, X_train,y_train,cv = 10)
print(np.mean(svc_score))

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(classification_report(y_test, y_pred))

#Decision Tree - Mean score = 81%- Report score = 79%
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

param_grid = {'criterion':['gini','entropy'],
              'max_depth':[1,2,3,4,5,6,7,8,9,10],
              'min_samples_split':[2,3,4,5,6,7,8,9,10,11,12,13,14,15],
              'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10]}
grid_search = GridSearchCV(tree,param_grid, cv = 10)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 9,
                              min_samples_split = 12, min_samples_leaf = 3)
tree_score = cross_val_score(tree, X_train, y_train, cv = 10)
print(np.mean(tree_score))

tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(classification_report(y_test,y_pred))