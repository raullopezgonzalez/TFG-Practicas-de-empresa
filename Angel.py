
import numpy as np
import pandas as pd
import scipy
#from scipy import interp
import matplotlib.pyplot as plt
# activate latex text rendering
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.feature_selection import SelectFwe, SelectPercentile, f_classif
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc, classification_report, recall_score
import shap
from itertools import compress



# #############################################################################
# Data IO and generation

# Import some data to play with
data = pd.read_csv('Thal_Features_mod_clean_covariates.csv', sep=',', dtype=np.float64)
field_names_list = data.drop(['condition','age','genotype'], axis=1).columns.values
CLBP_full = data.drop(['condition'], axis=1).values
CLBP_data = data.drop(['condition','age','genotype'], axis=1).values
CLBP_cond = data['condition'].values
CLBP_ages = data['age'].values
CLBP_geno = data['genotype'].values
CLBP_covs = np.concatenate((CLBP_ages.reshape(-1,1),CLBP_geno.reshape(-1,1)),axis=1)



n_samples, n_features = CLBP_data.shape

print('Loaded dataset including',str(n_samples),'samples with',str(n_features),'features')

# #############################################################################
# Classification and ROC analysis

# Split the dataset into training and test sets using a K fold approach

n_splits = 5
n_repeats = 1000

cv = RepeatedStratifiedKFold(n_splits,n_repeats,random_state=36851234)

# Run classifier with cross-validation and plot boundaries and ROC curves

#classifier = RandomForestClassifier(max_depth=None, n_estimators=1024, max_features="sqrt", oob_score=False)
classifier = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=4),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=3, max_features=0.9000000000000001, min_samples_leaf=12, min_samples_split=10, n_estimators=100, subsample=0.7500000000000001)),
    SelectFwe(score_func=f_classif, alpha=0.011),
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.7500000000000001, min_samples_leaf=9, min_samples_split=13, n_estimators=100)),
    Nystroem(gamma=0.4, kernel="additive_chi2", n_components=n_features),
    Nystroem(gamma=0.2, kernel="linear", n_components=n_features),
    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.05, min_samples_leaf=4, min_samples_split=12, n_estimators=100)
)


# Model Evaluation
performance = []
sensitivity = []
specificity = []

tprs = []   # True prediction errors
aucs = []   # Area under the curve
Ys = []
Zs = []

median_fpr = np.linspace(0, 1, 1000)

feat_importances = []

fig1 = plt.figure(figsize=(10, 10))
ax1 = fig1.add_subplot(111)
i = 0
for train, test in cv.split(CLBP_data, CLBP_cond):
  X_train = CLBP_data[train]
  Y_train = CLBP_cond[train]
  X_test = CLBP_data[test]
  Y_test = CLBP_cond[test]
  
  Ys.extend(Y_test) # Just a copy of the Y_test values in this void list
  # Regress out confoundings 
  # (Variable that influences both the dependent variable and independent variable)
  
  for idx in range(0,X_train.shape[1]):
    regr = linear_model.LinearRegression() 
    regr.fit(CLBP_covs[train], X_train[:,idx])    # CLBP_covs = np.concatenate((CLBP_ages.reshape(-1,1),CLBP_geno.reshape(-1,1)),axis=1)
    X_train[:,idx] = regr.predict(CLBP_covs[train]) - X_train[:,idx] + regr.intercept_
    X_test[:,idx] = regr.predict(CLBP_covs[test]) - X_test[:,idx] + regr.intercept_
  
  # Train the classifier
  try:
    classifier.fit(X_train, Y_train)
    # Compute ROC curve, area under the curve and SHAP values
    Z = classifier.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, Z[:, 1])   # Falpse positive and true positive rate
    roc_auc = auc(fpr, tpr) 
    
    shap_values = shap.TreeExplainer(classifier.named_steps['extratreesclassifier']).shap_values(X_train)   # SHapley Additive exPlanations
    # SHAP : impact of variables taking into account the interaction with other variables.
    
    print('Results for iteration ',str(i),' computed.')
    
    # Assign all computed values to their corresponding variables
    performance.append(classifier.score(X_test,Y_test))
    sensitivity.append(recall_score(Y_test, classifier.predict(X_test),pos_label=1))
    specificity.append(recall_score(Y_test, classifier.predict(X_test),pos_label=0))
    tprs.append(np.interp(median_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(roc_auc)
    Zs.extend(Z[:, 1])
    feat_importances.append(np.median(np.abs(shap_values[1]),axis=0))
    
  except:
    print('Error in iteration ',str(i))
    performance.append(float('nan'))
    sensitivity.append(float('nan'))
    specificity.append(float('nan'))
    tprs.append(float('nan'))
    aucs.append(float('nan'))
    Zs.extend(Y_test*float('nan'))
    feat_importances.append(float('nan'))

  i += 1

# Clean variables
index = ~np.isnan(performance)
clean_performance = list(compress(performance, index))
clean_sensitivity = list(compress(sensitivity, index))
clean_specificity = list(compress(specificity, index))
clean_tprs = list(compress(tprs, index))
clean_aucs = list(compress(aucs, index))
clean_feat_importances = list(compress(feat_importances, index))

# Summary
print('Accuracy',str(round(np.median(clean_performance),2)),'+',str(round(scipy.stats.median_absolute_deviation(clean_performance, scale=1.0),2)))
print('Sensitivity',str(round(np.median(clean_sensitivity),2)),'+',str(round(scipy.stats.median_absolute_deviation(clean_sensitivity, scale=1.0),2)))
print('Specificity',str(round(np.median(clean_specificity),2)),'+',str(round(scipy.stats.median_absolute_deviation(clean_specificity, scale=1.0),2)))

# Plot random ROC curve
ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)

# Plot median ROC curve with MAD
median_tpr = np.median(clean_tprs, axis=0)
median_tpr[-1] = 1.0
median_auc = auc(median_fpr, median_tpr)
MAD_auc = scipy.stats.median_absolute_deviation(clean_aucs, scale=1.0)
ax1.plot(median_fpr, median_tpr, color='b',
  label=r'Median ROC (AUC = %0.2f $\pm$ %0.2f)' % (median_auc, MAD_auc),
  lw=2, alpha=.8)

MAD_tpr = scipy.stats.median_absolute_deviation(clean_tprs, scale=1.0, axis=0)
tprs_upper = np.minimum(median_tpr + MAD_tpr, 1)
tprs_lower = np.maximum(median_tpr - MAD_tpr, 0)
ax1.fill_between(median_fpr, tprs_lower, tprs_upper, color='blue', alpha=.1,
  label=r'$\pm$ 1 MAD')

# Display ROC curves
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
ax1.tick_params(labelsize=16)
plt.xlabel('1-Specificity', fontsize=20)
plt.ylabel('Sensitivity', fontsize=20)
plt.title('ROC Curve', fontsize=22)
plt.legend(loc="lower right")
plt.show()

# Feature Importances
median_importances = np.median(clean_feat_importances, axis=0)
MAD_importances = scipy.stats.median_absolute_deviation(clean_feat_importances, scale=1.0, axis=0)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.bar(range(0,len(median_importances)),median_importances)
plt.xticks(np.arange(0,len(median_importances)), field_names_list, rotation='vertical')
plt.show()
