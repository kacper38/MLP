#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:50:58 2020

@author: kacper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
#%%
KATALOG_PROJEKTU = os.path.join(os.getcwd())
KATALOG_DANYCH = os.path.join(KATALOG_PROJEKTU,"archive")
KATALOG_WYKRESOW = os.path.join(KATALOG_PROJEKTU, "wykresy")
os.makedirs(KATALOG_WYKRESOW, exist_ok=True)


def load_data(data_path=KATALOG_DANYCH):
    csv_path = os.path.join(data_path, 'winequality-red.csv')
    return pd.read_csv(csv_path)

#def encode_quality(data):
#        data['quality'] = pd.Categorical(data['quality'].apply(\
#                            lambda x: 'low' if x <=6.5 \
#                                else 'good'))
#%%
if __name__ == "__main__":
    
    wines = load_data()
    #%%
    #podglad danych
    print(wines.head())
    
    #atrybuty
    print(wines.columns)
    
    #ogolne info
    print(wines.info())
    
    #sprawdzenie czy i ile jest niekompletnych danych
    print(wines.isnull().sum())
    
    #%%
    
    #jakosc wina i jego ilosc
    print(f'\n quality: \n{wines["quality"].value_counts(sort=False, bins=2)}')
    print(f'\n quality: \n{wines["quality"].value_counts(sort=False)}')
    
    
    #Added sulfites preserve freshness and protect wine from oxidation,
    #and unwanted bacteria and yeasts.
    print(f'\n sulphates: \n{wines["sulphates"].value_counts(sort=False,bins=5)}')
    #%%
    
    statistics = wines.describe()
    
    for el in statistics:
        print(f"{el}  mean= {statistics[el][1]:.3f},"
              f"  min= {statistics[el][4]:.3f}, max= {statistics[el][4]:.3f}\n")
    
    
    #%%
    
    sns.barplot(x='quality', y='alcohol', data=wines)
    plt.show()
    sns.barplot(x='quality', y='sulphates', data=wines)
    plt.show()
    sns.barplot(x='quality', y='citric acid', data=wines)
    plt.show()
    sns.barplot(x='quality', y='density', data=wines)
    plt.show()
    sns.barplot(x='quality', y='pH', data=wines)
    plt.show()
    sns.barplot(x='quality', y='chlorides', data=wines)
    plt.show()
    sns.barplot(x='quality', y='volatile acidity', data=wines)
    plt.show()
    sns.barplot(x='quality', y='total sulfur dioxide', data=wines)
    plt.show()
    """
    More alcohol better wine!
    Increased level of sulphates better quality
    
    Serious increase in citric acid in high quality wines
    
    not so much going on with the quality/density 
    slight decrease in pH for wines with better quality
    
    The better the wine the less chlorides it contains
    in general less volatile acidity better quality 
    
    And not so much knowledge from quality/total sulfur dioxide graph
    
    """
    #%% 
    #visualisation of correlations
    
    pd.plotting.scatter_matrix(wines[wines.columns], figsize=(24,24))
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'correlations.jpg'), dpi=300 ) 
    
    
    #%%
    wines.hist(bins=12, color='green', edgecolor='black',
               xlabelsize=6, ylabelsize=6,linewidth=0.8,
               grid=False, figsize=(14,14))
    plt.tight_layout(rect=(0.2, 0.2, 1.2, 1.2))
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'corr_2_hists.jpg'), dpi=300 )
    
    #%%
    correlation_matrix = wines.corr()
    for el in wines.columns:
        
        print(f"\nCorrelation with: {el}\n")
        print(f"{correlation_matrix[el].sort_values(ascending=True)}")
    
    
    """
    
    Strong correlations: < +/- 0.6 - 1>
    
        fixed acidity - pH              <negative>
        fixed acidity - density         <positive>
        fixed acidity - citric acid     <positive>
        
        volatile acidity - citric acid  <negative>
        
        citric acid - pH                <negative>
        
        free sulfur dioxide - total sulfur dioxide <positive>
        
    """
    
    sns.lineplot(data=wines,x='fixed acidity', y='pH' )
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'fix_acid_by_ph.jpg'), dpi=300 ) 
    plt.show()
    
    sns.lineplot(data=wines,x='fixed acidity', y='density' )
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'fix_aci_by_dens.jpg'), dpi=300 ) 
    plt.show()
    
    sns.lineplot(data=wines,x='fixed acidity', y='citric acid' )
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'fixed_aci_by_citricacid.jpg'), dpi=300 ) 
    plt.show()
    
    sns.lineplot(data=wines,x='volatile acidity', y='citric acid' )
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'v_aci_by_citr_aci.jpg'), dpi=300 ) 
    plt.show()
    
    sns.lineplot(data=wines,x='citric acid', y='pH' )
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'citr_aci_by_ph.jpg'), dpi=300 ) 
    plt.show()
    
    sns.lineplot(data=wines,x='free sulfur dioxide',
                 y='total sulfur dioxide')
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'free_sulf_diox_by_total_sulf_diox.jpg'),
                dpi=300 ) 
    plt.show()
    #%%
    
    #correlations in sns heatmap
    fig,ax = plt.subplots(figsize=(12,12))
    
    sns.heatmap(correlation_matrix, annot=True, ax=ax,
                linewidths=.1, fmt='.2', cmap='jet_r')
    fig.subplots_adjust(top=0.95)
    fig.suptitle('Wine attributes correlations Heatmap', fontsize=14)
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'correlations_map.jpg'), dpi=300 )
    
    
    #%%
    #based on tips in the dataset
    
    wines['quality'] = pd.Categorical(wines['quality'].apply(\
                            lambda x: 'low' if x <=6.5 \
                                else 'high'))
    
    print(wines['quality'].head())
    print(wines['quality'].value_counts())
    #%%
    # to show some statistics by wine type
    
    attributes = ['alcohol', 'sulphates', 'volatile acidity', 'pH', 'chlorides']
    
    low_info = wines[wines['quality'] == 'low'][attributes].describe()
    
    high_info = wines[wines['quality'] == 'high'][attributes].describe()
    print('*'*80,'\nlow quality wines:\n')
    print('\n',low_info)
    print('*'*80,'\n','high quality wines:')
    print('\n',high_info)
    print('*'*80)
    
    #%% frequency in wine
    wine_q  = wines['quality'].value_counts()
    wine_quality = (list(wine_q.index), list(wine_q.values))
    
    plt.bar(wine_quality[0],wine_quality[1], color='green')
    plt.xlabel('Quality')
    plt.ylabel('Number of observed')
    plt.savefig(os.path.join(KATALOG_WYKRESOW,
                        'quality_by_frequency_2_categories.jpg'),
                dpi=300 )
    
    plt.show()

    #%%
    
    #from strong correlations
    
    important_attr = ['quality','pH', 'density', 'sulphates',
                      'total sulfur dioxide', 'fixed acidity',
                      'citric acid','free sulfur dioxide']
    
    sns.pairplot(wines[important_attr], height=3, hue = 'quality',
                 aspect =2)
    plt.title('Wines attributes pair plots')
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'selected_correlations.jpg'), dpi=300 )
    
    plt.show()



#%%

#skalowanie danych
from sklearn.preprocessing import LabelEncoder

labele = LabelEncoder()

wines['quality'] = labele.fit_transform(wines['quality'])
#label encoder transformed quality labels from low, high to
# --> high became 0, low became 1

#%%
attributes = wines.drop('quality', axis = 1) 
features =  wines['quality']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(attributes, features,
                                        test_size = 0.2, random_state=131)

print('X_train shape',X_train.shape)
print('X_test shape',  X_test.shape)

#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# u mnie rozwiazaniem probelmu jest klasyfikacja, czyli wyznaczenie
# na podstawie danych czy wino bedzie wysokiej jakosci czy nie
#%%
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_squared_error as MSE

from sklearn.linear_model import SGDClassifier

def SGD(X_train,y_train,X_test,y_test):
    stochasticgd = SGDClassifier(loss='perceptron', penalty='l2')
    stochasticgd.fit(X_train, y_train)
    sgd_pred = stochasticgd.predict(X_test)
    
    mse_sgd = MSE(y_test, sgd_pred)
    rmse_sgd = mse_sgd**(1/2)
    
    print('SGDClassifier')
    print(classification_report(y_test, sgd_pred))
    print(f'accurracy= {accuracy_score(y_test, sgd_pred):.4f}')
    print(f'mse = {mse_sgd:.4f}')
    print(f'rmse = {rmse_sgd:.4f}')
    print('*'*70)

SGD(X_train,y_train,X_test,y_test)
#%%
from sklearn.tree import DecisionTreeClassifier

def DTC(X_train,y_train,X_test,y_test):
    dtc = DecisionTreeClassifier(max_depth=4, random_state=1,
                                 criterion='gini')
    
    dtc.fit(X_train,y_train)
    
    dtc_pred = dtc.predict(X_test)
    mse_dtc = MSE(y_test, dtc_pred)
    rmse_dtc = mse_dtc**(1/2)
    
    print('Decision Tree Classifier')
    print(classification_report(y_test, dtc_pred))
    print(f'accurracy= {accuracy_score(y_test, dtc_pred):.4f}')
    print(f'mse = {mse_dtc:.4f}')
    print(f'rmse = {rmse_dtc:.4f}')
    print('*'*70)

DTC(X_train,y_train,X_test,y_test)
#%%
from sklearn.neighbors import KNeighborsClassifier

def KNN(X_train,y_train,X_test,y_test):
    knn = KNeighborsClassifier(n_neighbors=8)   
    knn.fit(X_train,y_train)
    knn_pred = knn.predict(X_test)
    
    mse_knn = MSE(y_test, knn_pred)
    rmse_knn= mse_knn**(1/2)
    
    print('KNeighborsClassifierier')
    print(classification_report(y_test, knn_pred))
    print(f'accurracy= {accuracy_score(y_test, knn_pred):.4f}')
    print(f'mse = {mse_knn:.4f}')
    print(f'rmse =  {rmse_knn:.5f}')
    print('*'*70)

KNN(X_train,y_train,X_test,y_test)
#%%
from sklearn.linear_model import LogisticRegression

def LR(X_train,y_train,X_test,y_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    mse_lr = MSE(y_test, lr_pred)
    rmse_lr= mse_lr**(1/2)
    
    print('LogisticRegression')
    print(classification_report(y_test, lr_pred))
    print(f'accurracy= {accuracy_score(y_test, lr_pred):.4f}')
    print(f'mse = {mse_lr:.4f}')
    print(f'rmse = {rmse_lr:.4f}')
    print('*'*70)

LR(X_train,y_train,X_test,y_test)
#%%
from sklearn.svm import LinearSVC

def SVC(X_train,y_train,X_test,y_test):
    svc = LinearSVC(max_iter=10000)
    svc.fit(X_train, y_train)
    svc_pred = svc.predict(X_test)
    
    mse_svc= MSE(y_test, svc_pred)
    rmse_svc= mse_svc**(1/2)
    
    print('LinearSVC')
    print(classification_report(y_test, svc_pred))
    print(f'accurracy= {accuracy_score(y_test, svc_pred):.4f}')
    print(f'mse = {mse_svc:.4f}')
    print(f'rmse = {rmse_svc:.4f}')
    print('*'*70)
    
SVC(X_train,y_train,X_test,y_test)

#%%

from sklearn.ensemble import GradientBoostingClassifier

def GBC(X_train,y_train,X_test,y_test):
    gbc = GradientBoostingClassifier()
    
    gbc.fit(X_train,y_train)
    gbc_pred = gbc.predict(X_test)
    
    mse_gbc= MSE(y_test, gbc_pred)
    rmse_gbc= mse_gbc**(1/2)
    
    print('GradientBoostingClassifier')
    print(classification_report(y_test, gbc_pred))
    print(f'accurracy= {accuracy_score(y_test, gbc_pred):.4f}')
    print(f'mse = {mse_gbc:.4f}')
    print(f'rmse = {rmse_gbc:.4f}')
    print('*'*70)

GBC(X_train,y_train,X_test,y_test)


#%%

from xgboost import XGBClassifier

def XGB(X_train,y_train,X_test,y_test):

    xg_cl = XGBClassifier(objective='binary:logistic',
    n_estimators=10, seed=123)
    xg_cl.fit(X_train, y_train)
    
    xgcl_preds = xg_cl.predict(X_test)
    
    mse_xgcl= MSE(y_test, xgcl_preds)
    rmse_xgcl= mse_xgcl**(1/2)
    
    print('XGB')
    print(classification_report(y_test, xgcl_preds))
    print(f'accurracy= {accuracy_score(y_test, xgcl_preds):.4f}')
    print(f'mse = {mse_xgcl:.4f}')
    print(f'rmse = {rmse_xgcl:.4f}')
    print('*'*70)

XGB(X_train,y_train,X_test,y_test)
#%%



"""  https://tinyurl.com/y2jdf9n7
F1 score - F1 Score is the weighted average of Precision and Recall. 
 Therefore, this score takes both false positives and false negatives into account. 
 Intuitively it is not as easy to understand as accuracy, 
 but F1 is usually more useful than accuracy, 
 especially if you have an uneven class distribution

 F1 Score = 2*(Recall * Precision) / (Recall + Precision)


Accuracy - Accuracy is the most intuitive performance measure and it is simply
 a ratio of correctly predicted observation to the total observations. 
 One may think that, if we have high accuracy then our model is best. 
 Yes, accuracy is a great measure but only when you have symmetric datasets 
 where values of false positive and false negatives are almost same.
 
 Accuracy = TP+TN/TP+FP+FN+TN
 
 
Precision - Precision is the ratio of correctly predicted positive observations
 to the total predicted positive observations. 
 High precision relates to the low false positive rate.
 
 Precision = TP/TP+FP
 
 
Recall (Sensitivity) - Recall is the ratio of correctly predicted
 positive observations to the all observations in actual class - yes. 
 The question recall answers is: Of all the passengers that truly survived,
 how many did we label?
 
 Recall = TP/TP+FN
"""

from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB

scoring = ['accuracy','f1', 'recall','precision']

modele = [GradientBoostingClassifier(),LinearSVC(max_iter=10000),
          LogisticRegression(),GaussianNB(),
          KNeighborsClassifier(),XGBClassifier(),
          DecisionTreeClassifier(),SGDClassifier(penalty='l2',loss='log')]


def cvs (modele,X_train,y_train,X_test,y_test):
    acc_dict  = {}
    f1_dict   = {}
    rec_dict  = {}
    prec_dict = {}
    
    for model in modele:
        clf = model.fit(X_train, y_train)
        clf_scores = cross_validate(clf, X_train, y_train,
                              scoring=scoring, cv=10)
        
        acc_dict[str(model)[:-2]] = clf_scores['test_accuracy']
        f1_dict[str(model)[:-2]] = clf_scores['test_f1']
        rec_dict[str(model)[:-2]] = clf_scores['test_recall']
        prec_dict[str(model)[:-2]] = clf_scores['test_precision']
    return acc_dict,f1_dict, rec_dict, prec_dict

stats = cvs(modele,X_train,y_train,X_test,y_test)

models = list(stats[0].keys())
#sml stands for scores_means_list
sml = [np.array(list(x.values())).mean(axis=1) for x in stats]


sp = sns.barplot(x=sml[0], y=models).set_title('accuracy_means')
plt.savefig(os.path.join(KATALOG_WYKRESOW,'accuracy_between_models.jpg'), dpi=300 )
plt.show()

sns.barplot(x=sml[2], y=models).set_title('recall_means')
plt.savefig(os.path.join(KATALOG_WYKRESOW,'recall_between_models.jpg'), dpi=300 )
plt.show()

sns.barplot(x=sml[3], y=models).set_title('precision_means')
plt.savefig(os.path.join(KATALOG_WYKRESOW,'precision_between_models.jpg'), dpi=300 )
plt.show()

sns.barplot(x=sml[1], y=models).set_title('f1_means')
plt.savefig(os.path.join(KATALOG_WYKRESOW,'f1_betweem_models.jpg'), dpi=300 )
plt.show()
#%%


for i, scorer in enumerate(['accuracy','f1','recall','precision']):

    print(f"{scorer} := biggest_mean -> {sml[i].max():.4f}"
          f" model := {models[np.argmax(sml[i])]}\n")
    

"""
as the winner of f1 score is the GradientBoostingClassifier, 
and f1 score takes both recall and precision into account,
and it also wins in the accuracy test.
As I think here false negatives and false negatives are of crucial role,
remember high quality wines otnumber low quality by a big margin, 
there are circa 6 times more low quality wines than high

So the double Winning GradientBoostingClassifier is a go to there/
"""

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


X, y = attributes, features

model = GradientBoostingClassifier()
default_params = model.get_params()
print(default_params)#aby wiedziec co mozna zmieniac, pozniej i tak musialem czytac dokumentacje, aby wiedziec co jest czym..
#%%
#LICZY OKOLO GODZINE, poniezej moje wyniki, 
#nie skorzystalem z grid search, ponieważ za duza 'mapa' parametrow
#i szukanie trwaloby wiele lat
parameters = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.608, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10,100,1000,10000]
    }

number_combinations = 1
for i in parameters:
    number_combinations *= len(i)


cv_random = RandomizedSearchCV(estimator = GradientBoostingClassifier(),
        param_distributions=parameters, n_iter=100,
        scoring='f1',
        n_jobs=-1,
        refit=True,
        cv=10, verbose=1,
        return_train_score=True)
    
search = cv_random.fit(X_train,y_train)

best_score = search.best_score_
best_params = search.best_params_

'''
best_scorew = 0.9457450478391032
best_paramsw = {'subsample': 1.0, 
                'n_estimators': 1000,
                'min_samples_split': 0.2090909090909091,
                'min_samples_leaf': 0.1, 
                'max_features': 'log2', 
                'max_depth': 5, 
                'loss': 'deviance', 
                'learning_rate': 0.1, 
                'criterion': 'friedman_mse'}
'''
"""
    1000 fits is about 44 minutes,
    1000/4cores*10sec(per_model)*60
"""
#%%

print(search.best_score_)
print(search.best_params_)


#%%
from sklearn.model_selection import  learning_curve, validation_curve

#plots generated as here -> https://chrisalbon.com/machine_learning/model_evaluation/plot_the_validation_curve/

param_range=np.arange(1,250,2)

train_scores, val_scores = validation_curve(
    GradientBoostingClassifier().set_params(**best_params), 
    attributes, features,
    param_name='n_estimators',
    param_range=param_range,
    cv = 10,
    scoring='f1',
    n_jobs=-1)


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, val_mean, label="Cross-validation score", color="dimgrey")

plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, color="gainsboro")


plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Trees")
plt.ylabel("f1 Score")
plt.tight_layout()
plt.legend(loc="best")
plt.savefig(os.path.join(KATALOG_WYKRESOW,'validation_curve.jpg'), dpi=300 )
plt.show()

''' Widac ze dla duzej ilosci drzew model zayczna byc zbyt dobrze dopasowany 
do zbioru treningowego
'''


#dorobic learning curve pojedyncza
#%%
train_sizes, train_scores, test_scores =\
    learning_curve(GradientBoostingClassifier().set_params(**best_params), 
                   X, y,
                   cv=10,
                   scoring='f1',
                   n_jobs=-1, 
                   train_sizes=np.linspace(0.01, 1.0, 50))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("f1 Score"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(KATALOG_WYKRESOW,'learning_curve.jpg'), dpi=300 )
plt.show()


#%%




gbcc = GradientBoostingClassifier().set_params(**best_params)
    
gbcc.fit(X_train,y_train)
gbc_pred = gbcc.predict(X_test)
    
mse_gbc= MSE(y_test, gbc_pred)
rmse_gbc= mse_gbc**(1/2)
    
print('GradientBoostingClassifier')
print(classification_report(y_test, gbc_pred))
print(f'accurracy= {accuracy_score(y_test, gbc_pred):.4f}')
print(f'mse = {mse_gbc:.4f}')
print(f'rmse = {rmse_gbc:.4f}')
print('*'*70)

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_test, gbc_pred))



#%%
from sklearn.model_selection import cross_val_score


scoresgbc = cross_val_score(gbcc, 
                            X_train, y_train, 
                            scoring=None, 
                            cv=10,
                            n_jobs=-1)



print(f'cross validation score: {scoresgbc.mean():.4f}')

print(f'Accuracy: {np.mean(scoresgbc):.4f},  std: {np.std(scoresgbc):.4f}')

y_pred_train = gbcc.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train)
print(f'accuracy on training set: {acc_train:.4f}')

y_pred_test = gbcc.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)
print(f'accuracy on test set: {acc_test:.4f}\n')

print(classification_report(y_test, y_pred_test))


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,y_pred_test))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()


#%%
#at last lets get confusion matrix plot
from sklearn.metrics import plot_confusion_matrix

labels=['true negatives','false positives','false negatives','true positives']

disp = plot_confusion_matrix(gbcc, X_test, y_test,
                             cmap=plt.cm.Blues,
                             normalize='all')

disp.ax_.set_title('normalized confusion matrix plot')
print(disp.confusion_matrix)
plt.show()
    
    
'''So I ended up with accuracy on the test set of 0.9281,
 and cv score of 0.9424(f1), and 0.8991(accuracy)
got 0.85 of true positives(271) which is not bad 
considering small size of data,
it seems that algorithm has efficiently learned,
which are the best chemical traits, to select a good wine,
what a shame they don't put those on wine labels...
maybe, next step, -> indentify wine by etiquette ->
connect with database of those chemical properties,
-> return answer.
'''    
    
    
    


