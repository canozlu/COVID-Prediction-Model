import pandas as pd
import glob
import re
import os
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

filenames = glob.glob('*')
for name in filenames:
    if re.search('\.',name):
        filenames.remove(name)
    else:
        None
        
all_data_names = {}                             #Grouping the collection of COVID-19 datasets I have  from Kaggle together so I can open/combine them easily
for name in filenames:
    for root, dirs, files in os.walk(name):
        all_data_names[name] = files
        
        
ein_df = pd.read_csv('einstein/' + all_data_names['einstein'][0])               #This dataset is from Albert Einstein Hospital in Brazil (Hospital Israelita Albert Einstein), dataset is available on Kaggle, https://www.kaggle.com/einsteindata4u/covid19
ein_df.head()
ein_df.info()                           #A lot of NaN values discovered, unfortunately

ein_df.corr()[ein_df.corr()> 0.8]                                       #Checking for highly correlated lab variables
ein_df = ein_df.drop('phosphor', axis=1)                            #Something seems wrong with this lab value, correlates with unrelated labs, so removed

def correlation(dataset, threshold):                                #This function deletes one of the two highly correlated (above threshold) variables' columns from each pair (e.g. hematocrit & hemoglobin), I manually inspected beforehand.
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):      #second part allows unnecessary extra deletion
                colname = corr_matrix.columns[i] # getting the name of column
                colname2 = corr_matrix.columns[j] # getting the name of column 2
                if dataset[dataset[colname].isnull()][colname2].notnull().sum() < 100:      #Extra: This may throw an error but works to check that we are not losing more data by removing columns (checks if the correlated column has a value)
                    col_corr.add(colname)
                    if colname in dataset.columns:
                        del dataset[colname] # deleting the column from the dataset

    print(dataset)

correlation(ein_df,0.84)    #error, but works, may inspect later

corr_metrics = ein_df.corr()
corr_metrics.style.background_gradient()        #checking
ein_df.info()


ein_df = ein_df.dropna(thresh=100, axis='columns') #drop columns with less than 100 non-missing values

for i in range(18,35):
    for index, row in ein_df.iterrows():
        if row[i] == 'detected' and row['sars_cov_2_exam_result'] == 'positive':            #removed cases with concurrent infections
            ein_df.drop(ein_df.index[index], inplace=True)

listofothervirus = ein_df.columns[range(18,35)]
ein_df.drop(listofothervirus, axis = 1, inplace=True)           #removed other viral tests from dataset

for index, row in ein_df.iterrows():
        if (row['influenza_b_rapid_test'] == 'detected' or row['influenza_a_rapid_test'] == 'detected') and row['sars_cov_2_exam_result'] == 'positive':
            ein_df.drop(ein_df.index[index], inplace=True)

for index, row in ein_df.iterrows():
        if (row['strepto_a'] == 'detected') and row['sars_cov_2_exam_result'] == 'positive':             #removed cases with concurrent infections
            ein_df.drop(ein_df.index[index], inplace=True)
            
ein_df.drop(['strepto_a','influenza_b_rapid_test','influenza_a_rapid_test'], axis = 1, inplace=True) #removed other viral tests from dataset


import matplotlib.pyplot as plt


ein_df['patient_age_quantile'].plot.kde()                       #Important: Test the model after stratifying ccording to the age of the patient later
plt.show()



ein_df_removed_missing = ein_df.dropna(thresh=20, axis='rows')              #Removed rows/patients with too many missing lab values so the model works better. Con: This means we end up with patients who have had a lot of labwork


(ein_df_removed_missing['sars_cov_2_exam_result']== 'positive').sum()
(ein_df_removed_missing['sars_cov_2_exam_result']== 'negative').sum()           #checking the amount of positive & negative test results in the dataset, will balance so the model is not skewed. Con: Smaller sample size


positive_results = ein_df_removed_missing[ein_df_removed_missing['sars_cov_2_exam_result'] == 'positive']
negative_results = ein_df_removed_missing[ein_df_removed_missing['sars_cov_2_exam_result'] == 'negative']


negative_results = negative_results.sample(n=len(positive_results),random_state=37)         #randomly sampling the COVID negative group to equalize the number to the COVID positive group


ein_df_feq = pd.concat([negative_results, positive_results]

features = ein_df_feq.columns[range(6,36)]
labels = ein_df_feq.columns[range(2,6)]

X_feq,y_feq = ein_df_feq[features], ein_df_feq[labels]              #separating the data into features and labels. Features: Lab tests. Labels: COVID test result, ICU admission (not modeled here)

imputer = KNNImputer()                                  #Imputing missing values with KNNeighbors method, default n_neighbors used
X_feqimp = imputer.fit_transform(X_feq)

pca = PCA()                                             #PCA fitted to reduce number of features
pca.fit(X_feqimp)

exp_variance = pca.explained_variance_ratio_
print(len(exp_variance)
print(exp_variance)

fig, ax = plt.subplots()                            #Scree plot checked to reduce dimensionality, saw a clear elbow after 5 factors.
ax.bar(x=range(30),height=exp_variance)
ax.set_xlabel('Principal Component #')                       
                       
n_components = 5                    
                      
pca = PCA(n_components, random_state=37)
pca.fit(X_feqimp)
pca_projection_feq = pca.transform(X_feqimp)


#Splitting the dataset into training and test for the model, default split used. No cross validation done here, may include cross-val with KFold later
train_features_feq, test_features_feq, train_labels_feq, test_labels_feq = train_test_split(pca_projection_feq, y_feq['sars_cov_2_exam_result'], random_state=37)


tree = DecisionTreeClassifier(random_state=37)                              #Comparing different classifier models
logreg = LogisticRegression(random_state=37)                                #Initializing each model type
knn = KNeighborsClassifier()

classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]
vc = VotingClassifier(estimators=classifiers)

rf = RandomForestClassifier(n_estimators=25, random_state=37)

tree.fit(train_features_feq,train_labels_feq)                               #Fitting each model type on the training dataset
logreg.fit(train_features_feq,train_labels_feq)
knn.fit(train_features_feq,train_labels_feq)

vc.fit(train_features_feq,train_labels_feq)

rf.fit(train_features_feq,train_labels_feq)



fe_res_pred_tree = tree.predict(test_features_feq)                          #Predicting with the test dataset with each model type
fe_res_pred_logreg = logreg.predict(test_features_feq)
fe_res_pred_knn = knn.predict(test_features_feq)

fe_res_pred_vc = vc.predict(test_features_feq)

fe_res_pred_rf = rf.predict(test_features_feq)


print('tree:' + classification_report(test_labels_feq,fe_res_pred_tree))
print('logreg:' + classification_report(test_labels_feq,fe_res_pred_logreg))
print('knn:' + classification_report(test_labels_feq,fe_res_pred_knn))

print('vc:' + classification_report(test_labels_feq,fe_res_pred_vc))

print('rf:' + classification_report(test_labels_feq,fe_res_pred_rf))            #Random Forest Classifier produced the best results with 0.85 accuracy (with great accuracy in predicting positive and negative results)





