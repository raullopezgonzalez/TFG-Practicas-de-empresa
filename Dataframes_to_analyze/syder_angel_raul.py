import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib
#from matplotlib import pyplot as plt
#import os
#import statistics 
#from src.utils import plot_scatter, plot_silhouette
#from sklearn.cluster import KMeans
#from scipy.cluster.hierarchy import dendrogram, linkage
#from sklearn.cluster import AgglomerativeClustering
#from sklearn.metrics import silhouette_score
#import matplotlib.cm as cm
#from sklearn.preprocessing import StandardScaler


##############################################################################
##############################################################################

#import scipy
#from scipy import interp

#from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.feature_selection import SelectFwe, SelectPercentile, f_classif
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc, recall_score
#from sklearn.metrics import classification_report
#import shap
#from itertools import compress

##############################################################################
##############################################################################


#r'C:\Users\riul0\Desktop\TFG_Empresa\WORK\Dataframes_to_analyze\


data = pd.read_csv(r'C:\Users\riul0\Desktop\TFG_Empresa\WORK\Dataframes_to_analyze\Dataframe_no_kinematics_no_zeros_paired_sorted.csv', sep=',')
Dataframe = data.rename(columns = {"Unnamed: 0" :  "Patients"} )
Dataframe = Dataframe.set_index('Patients')

list_patients_data = []
for l in Dataframe.index:
     list_patients_data.append(l)
     
list_patients_data_M_DR = []
for i in range(len(list_patients_data)):
    if list_patients_data[i][0:4] == 'M_DR':
        list_patients_data_M_DR.append(list_patients_data[i])
     
Labels = pd.read_excel(r'C:\Users\riul0\Desktop\TFG_Empresa\WORK\Dataframes_to_analyze\Base datos clinica clustering.xlsx')
drop_columns = []
for i in Labels.columns[13:]:
    drop_columns.append(i)
drop_axis = []
Labels.drop(drop_columns, inplace=True, axis=1)
Labels = Labels[:40]

list_patients_label = []
for j in Labels.ID:
    list_patients_label.append(j)
    
new_list_patients_label = list_patients_label[0:34]    

Lista_no_encontrados = []
counter = 0
for i in list_patients_data_M_DR:
    if i[5:] in new_list_patients_label:
        continue
    else:
        Lista_no_encontrados.append(i)
        
Labels_2 = pd.read_excel(r'C:\Users\riul0\Desktop\TFG_Empresa\WORK\Dataframes_to_analyze\Pacientes_no_encontrados_Raul.xlsx')
drop_columns = []
for i in Labels_2.columns[13:]:
    drop_columns.append(i)
drop_axis = []
Labels_2.drop(drop_columns, inplace=True, axis=1)

list_patients_label_2 = []
for j in Labels_2.ID:
    list_patients_label_2.append(j[-15:])
    
    
Lista_no_encontrados_2 = []
counter = 0
for i in Lista_no_encontrados:
    if i in list_patients_label_2 :
        continue
    else:
        Lista_no_encontrados_2.append(i)
        
##############################################################################
##############################################################################

#r'C:\Users\riul0\Desktop\TFG_Empresa\WORK\Dataframes_to_analyze\

data = pd.read_csv(r'C:\Users\riul0\Desktop\TFG_Empresa\WORK\Dataframes_to_analyze\Dataframe_no_kinematics_no_zeros_paired_sorted.csv', sep=',')
Dataframe = data.rename(columns = {"Unnamed: 0" :  "Patients"} )
Patients = Dataframe["Patients"].to_list()
Patients_R = []
for i in Patients:
    if i[0:4] == 'M_DR':
        Patients_R.append(i[-10:])


Labels = pd.read_excel(r'C:\Users\riul0\Desktop\TFG_Empresa\WORK\Dataframes_to_analyze\Base datos clinica clustering.xlsx')
drop_columns = []
for i in Labels.columns[13:]:
    drop_columns.append(i)
drop_axis = []
Labels.drop(drop_columns, inplace=True, axis=1)
Labels = Labels[:34]

Labels_2 = pd.read_excel(r'C:\Users\riul0\Desktop\TFG_Empresa\WORK\Dataframes_to_analyze\Pacientes_no_encontrados_Raul.xlsx')
drop_columns = []
for i in Labels_2.columns[13:]:
    drop_columns.append(i)
drop_axis = []
Labels_2.drop(drop_columns, inplace=True, axis=1)
ID_labels_2 = Labels_2['ID'].to_list()

ID_labels_2_new = []
for i in ID_labels_2:
    ID_labels_2_new.append(i[-10:])
    
Labels_2.drop('ID', axis = 1, inplace = True)
Labels_2['ID'] = ID_labels_2_new

databases_label = [Labels, Labels_2]
Labels_both = pd.concat(databases_label)

Labels_both = Labels_both.set_index('ID')
Labels_sorted = Labels_both.loc[Patients_R]

asia_drop = Labels_sorted['ASIA'].to_list()
nl_drop =  Labels_sorted['NIVEL LESION'].to_list()

# Take care of minor d 

d_minor = []
for i in range(0,len(asia_drop)):
    if asia_drop[i] == 'd':
        d_minor.append(i)
        
##############################################################################
##############################################################################        

index_asia_drop = []
for i in range(0,len(asia_drop)):
    if (type(asia_drop[i]) == np.float) or (asia_drop[i] == 'NS'):
        index_asia_drop.append(i)
index_nl_drop = []
for i in range(0,len(nl_drop)):
    if (type(nl_drop[i]) == np.float) or (nl_drop[i] == 'NS'):
        index_nl_drop.append(i)

index_to_drop = index_asia_drop = index_nl_drop

patients_to_drop = []
for i in index_to_drop:
    patients_to_drop.append(Patients_R[i])

for i in patients_to_drop:
    Labels_sorted = Labels_sorted.drop(i)
    
Dataframe = Dataframe.set_index('Patients')
for i in patients_to_drop:
    m_dr = 'M_DR_' + i
    m_iz = 'M_IZ_' + i
    Dataframe = Dataframe.drop(m_dr)
    Dataframe = Dataframe.drop(m_iz)   
    
# Now we change d value from L077M1NAAA to D

Labels_sorted['ASIA'].loc[['L077M1NAAA']] = np.str('D')    
    
# Drop L2 Patient

Index_L = Labels_sorted.index[Labels_sorted['NIVEL LESION'] == 'L2'].tolist()
m_dr = 'M_DR_' + Index_L[0]
m_iz = 'M_IZ_' + Index_L[0]
Dataframe = Dataframe.drop(m_dr)
Dataframe = Dataframe.drop(m_iz)
Labels_sorted = Labels_sorted.drop(Index_L[0])


# Drop Nan patient

for i in data.columns:
    if (data[i].isnull().values.any() == True):
        index = data[i].index[data[i].apply(np.isnan)]
    else:
        continue

index = index[0][-10:]
m_dr_index = 'M_DR_' + index
m_iz_index = 'M_IZ_' + index
data = data.drop(m_dr_index)
data = data.drop(m_iz_index)
Labels_sorted = Labels_sorted.drop(index)

    
##############################################################################
##############################################################################
    

data = Dataframe
field_names_list = data.columns.values
field_names_list = field_names_list

Patient_data_aux = data.values
Patient_data = []
for idx in range(0,len(Patient_data_aux)):
    Patient_data.append(Patient_data_aux[idx,0:])
Patient_data = np.array(Patient_data)

field_names_R = list('R_' + field_names_list)
field_names_L = list('L_' + field_names_list)
field_names_all = field_names_R + field_names_L

Patient_data_all = Patient_data.reshape(int(Patient_data.shape[0]/2), int(2*Patient_data.shape[1]))

n_samples, n_features = Patient_data_all.shape

n_rows, n_columns = Labels_sorted.shape

Head_neck_diaf = ['C1', 'C2','C3'] # Group 0
# Upper_limbs = ['C4', 'C5','C6', 'C7', 'C8'] #
C_4 = 'C4' # 1
C_5 = 'C5' # 2
C_6 = 'C6' # 3
C_7 = 'C7' # 4
C_8 = 'C8' # 5
Pecto_muscles = ['D1', 'D2','D3', 'D4', 'D5', 'D6'] # Group 6
Abdo_muscles = [ 'D7', 'D8', 'D9','D10', 'D11', 'D12'] # Group 7
More_than_one = [ 'C3-D4','D3-D4' ] # Group 10

Labels_by_order = ['Head_neck_diaf', 'C_4', 'C_5', 'C_6', 'C_7', 'C_8', 'Pecto_muscles', 'Abdo_muscles', 
                   'Almost_whole_colum']

Labels_sorted['NIVEL LESION V2'] = np.nan # Creation/Reset


for i in range(0,len(list(Labels_sorted.index))):
    if (Labels_sorted['NIVEL LESION'].loc[list(Labels_sorted.index)[i]] in Head_neck_diaf) :
        Labels_sorted['NIVEL LESION V2'].loc[[list(Labels_sorted.index)[i]]] = 0
    if (Labels_sorted['NIVEL LESION'].loc[list(Labels_sorted.index)[i]] == C_4) :
        Labels_sorted['NIVEL LESION V2'].loc[[list(Labels_sorted.index)[i]]] = 1
    if (Labels_sorted['NIVEL LESION'].loc[list(Labels_sorted.index)[i]] == C_5) :
        Labels_sorted['NIVEL LESION V2'].loc[[list(Labels_sorted.index)[i]]] = 2
    if (Labels_sorted['NIVEL LESION'].loc[list(Labels_sorted.index)[i]] == C_6) :
        Labels_sorted['NIVEL LESION V2'].loc[[list(Labels_sorted.index)[i]]] = 3
    if (Labels_sorted['NIVEL LESION'].loc[list(Labels_sorted.index)[i]] == C_7) :
        Labels_sorted['NIVEL LESION V2'].loc[[list(Labels_sorted.index)[i]]] = 4
    if (Labels_sorted['NIVEL LESION'].loc[list(Labels_sorted.index)[i]] == C_8) :
        Labels_sorted['NIVEL LESION V2'].loc[[list(Labels_sorted.index)[i]]] = 5
    if (Labels_sorted['NIVEL LESION'].loc[list(Labels_sorted.index)[i]] in Pecto_muscles) :
        Labels_sorted['NIVEL LESION V2'].loc[[list(Labels_sorted.index)[i]]] = 6
    if (Labels_sorted['NIVEL LESION'].loc[list(Labels_sorted.index)[i]] in Abdo_muscles) :
        Labels_sorted['NIVEL LESION V2'].loc[[list(Labels_sorted.index)[i]]] = 7
    if (Labels_sorted['NIVEL LESION'].loc[list(Labels_sorted.index)[i]] in More_than_one) :
        Labels_sorted['NIVEL LESION V2'].loc[[list(Labels_sorted.index)[i]]] = 10

Labels_sorted['NIVEL LESION V2'] = Labels_sorted['NIVEL LESION V2'].astype(int)


number_repetitions_per_class = Labels_sorted.pivot_table(index=['NIVEL LESION V2'], aggfunc='size')
#print(number_repetitions_per_class) # All all right =)

##############################################################################
##############################################################################


n_splits = 2
n_repeats = 1

cv = RepeatedStratifiedKFold(n_splits,n_repeats,random_state=36851234)

classifier = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=4),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=3, max_features=0.9000000000000001, min_samples_leaf=12, min_samples_split=10, n_estimators=100, subsample=0.7500000000000001)),
    SelectFwe(score_func=f_classif, alpha=0.011),
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.7500000000000001, min_samples_leaf=9, min_samples_split=13, n_estimators=100)),
    Nystroem(gamma=0.4, kernel="additive_chi2", n_components=n_features),
    Nystroem(gamma=0.2, kernel="linear", n_components=n_features),
    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.05, min_samples_leaf=4, min_samples_split=12, n_estimators=100)
)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
n_classes = len(Labels_sorted['NIVEL LESION V2'].unique())







# Model Evaluation
performance = []
sensitivity = []
specificity = []

tprs = []   # True prediction errors
aucs = []   # Area under the curve
Ys = []
Zs = []

median_fpr = np.linspace(0, 1, 1000)

#feat_importances = []

#fig1 = plt.figure(figsize=(10, 10))
#ax1 = fig1.add_subplot(111)
i = 0

Labels_sorted_nl2 = Labels_sorted['NIVEL LESION V2'].values

for train, test in cv.split(Patient_data_all, Labels_sorted_nl2): 
    X_train = Patient_data_all[train]
    Y_train = Labels_sorted_nl2[train]
    X_test = Patient_data_all[test]
    Y_test = Labels_sorted_nl2[test]
    
    Ys.extend(Y_test) 

try:
    print(X_train)
    print(Y_train)
    classifier.fit(X_train, Y_train)
   
    # Compute ROC curve, area under the curve and SHAP values
    Z = classifier.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, Z[:, 1])   # Falpse positive and true positive rate
    roc_auc = auc(fpr, tpr) 
   
    
    #shap_values = shap.TreeExplainer(classifier.named_steps['extratreesclassifier']).shap_values(X_train)   # SHapley Additive exPlanations
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
    #feat_importances.append(np.median(np.abs(shap_values[1]),axis=0))

    
except:
    print('Error in iteration ',str(i))
    performance.append(float('nan'))
    sensitivity.append(float('nan'))
    specificity.append(float('nan'))
    tprs.append(float('nan'))
    aucs.append(float('nan'))
    Zs.extend(Y_test*float('nan'))
    #feat_importances.append(float('nan'))

    i += 1
