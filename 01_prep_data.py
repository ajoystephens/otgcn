# LIBRARIES
import networkx as nx
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,r2_score, roc_auc_score, confusion_matrix

from sklearn.preprocessing import normalize
import scipy
from scipy.spatial import distance
from scipy import sparse

import ABIDEParser as Reader


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
SEX_BOUNDARY = 0.7 # this file splits by the percent male at accquisition site
SIMILARITY_THRESHOLD = 0.65

print('Begin Data Prep Script')
print(f'SEX_BOUNDARY: {SEX_BOUNDARY}')
print(f'SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}')

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# METHODS
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def normalizeMinMax(x,minimum,maximum,invalid_value=0,shift=0):
    # x = pd.to_numeric(df['FIQ']) 
    x += shift
    minimum += shift
    maximum += shift
    
    isValid = (minimum<=x) & (x<=maximum)
    x[isValid] = (x-minimum)/(maximum-minimum)
    x[np.invert(isValid)] = invalid_value

    return(x)

# get site summary
def get_site_summary(data):
    site_summary = data.groupby(['site_id']).size().reset_index()
    site_summary = site_summary.rename(columns={0:'site_total'})
    temp_male = data[data['is_male']==1].groupby(['site_id']).size().reset_index()
    temp_male = temp_male.rename(columns={0:'site_male'})
    site_summary = site_summary.join(temp_male.set_index('site_id'), on='site_id')
    
    temp = data[data['is_autism']==1].groupby(['site_id']).size().reset_index()
    temp = temp.rename(columns={0:'site_autism'})
    site_summary = site_summary.join(temp.set_index('site_id'), on='site_id')

    data_total = site_summary['site_total'].sum()

    site_summary['p_site'] = (site_summary['site_total']/data_total).round(2)
    site_summary['p_male'] = (site_summary['site_male']/site_summary['site_total']).round(2)

    site_summary = site_summary.sort_values(by=['site_total'],ascending=False).fillna(0)
    return(site_summary)


def split_groups(data,sex_boundary = 0.7):
    group_conditions=[
        (data['p_male']>=sex_boundary),
        (data['p_male']<sex_boundary),
    ]
    
    group_values = [1, 2,]

    data['group'] = np.select(group_conditions, group_values, default=0)
    return(data)

def getAdjacencyMatrix(img_vectors,threshold):
    # Calculate all pairwise distances
    distv = distance.pdist(img_vectors, metric='correlation')
    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    # Get affinity from similarity matrix
    adj = np.exp(- dist ** 2 / (2 * sigma ** 2))

    # remove edges that dont meet the similarity threshold
    adj[adj < threshold] = 0
    
    # remove self loops (GCN typically adds these later but expects to start without
    adj = adj - np.eye(adj.shape[0])
    
    return(adj)
def getGraphInfoString(node_features,has_label,label,prot,adj):
    feat_cnt = node_features.shape[1]
    node_cnt = node_features.shape[0]
    labeled_node_cnt = sum(has_label)
    label_cnt = sum(label)
    male_node_cnt = sum(prot)
    male_label_node_cnt = sum(np.logical_and(prot,has_label))
    dog_person_node_cnt = sum(prot)
    dog_person_label_node_cnt = sum(np.logical_and(prot,has_label))
    edge_cnt = np.count_nonzero(adj)/2

    info = '\n\t'+str(node_cnt)+' nodes '
    info += '\n\t'+str(feat_cnt)+' node features '
    info += '\n\t'+str(labeled_node_cnt)+' labeled nodes '
    info += '\n\t'+str(edge_cnt)+' edges '
    info += '\n\t'+str(label_cnt)+' true label ('+str(label_cnt/labeled_node_cnt)+'p of labeled nodes)'
    info += '\n\t'+str(male_node_cnt)+' true prot ('+str(male_node_cnt/node_cnt)+'p of nodes)'
    info += '\n\t'+str(male_label_node_cnt)+' labeled true prot ('+str(male_label_node_cnt/labeled_node_cnt)+'p of labeled nodes)'

    return(info)


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# GET ABIDE DATA
# using parisot's reader
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
print('Retrieving ABIDE data....')
subject_IDs = Reader.get_ids()
labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
df = pd.DataFrame.from_dict(labels,orient='index',columns=['DX_GROUP'])

features = ['SEX','SITE_ID','FIQ','PIQ','VIQ',
            'FIQ_TEST_TYPE',
            'VIQ_TEST_TYPE',
            'PIQ_TEST_TYPE',
            'AGE_AT_SCAN',
            'ADI_R_SOCIAL_TOTAL_A',
            'ADI_R_VERBAL_TOTAL_BV',
            'ADI_RRB_TOTAL_C',
            'ADI_R_ONSET_TOTAL_D',
            'ADI_R_RSRCH_RELIABLE',
            'ADOS_MODULE',
            'ADOS_TOTAL',
            'ADOS_COMM',
            'ADOS_SOCIAL',
            'ADOS_STEREO_BEHAV',
            'ADOS_RSRCH_RELIABLE',
            'ADOS_GOTHAM_SOCAFFECT',
            'ADOS_GOTHAM_RRB',
            'ADOS_GOTHAM_TOTAL',
            'ADOS_GOTHAM_SEVERITY',
            'SRS_VERSION',
            'SRS_RAW_TOTAL',
            'SRS_AWARENESS',
            'SRS_COGNITION',
            'SRS_COMMUNICATION',
            'SRS_MOTIVATION',
            'SRS_MANNERISMS',
            'SCQ_TOTAL',
            'AQ_TOTAL',
            # 'COMORBIDITY',
            # 'CURRENT_MED_STATUS', come back
            # 'MEDICATION_NAME',
            'OFF_STIMULANTS_AT_SCAN',
            'VINELAND_RECEPTIVE_V_SCALED',
            'VINELAND_EXPRESSIVE_V_SCALED',
            'VINELAND_WRITTEN_V_SCALED',
            'VINELAND_COMMUNICATION_STANDARD',
            'VINELAND_PERSONAL_V_SCALED',
            'VINELAND_DOMESTIC_V_SCALED',
            'VINELAND_COMMUNITY_V_SCALED',
            'VINELAND_DAILYLVNG_STANDARD',
            'VINELAND_INTERPERSONAL_V_SCALED',
            'VINELAND_PLAY_V_SCALED',
            'VINELAND_COPING_V_SCALED',
            'VINELAND_SOCIAL_STANDARD',
            'VINELAND_SUM_SCORES',
            'VINELAND_ABC_STANDARD',
            'VINELAND_INFORMANT',
            'WISC_IV_VCI',
            'WISC_IV_PRI',
            'WISC_IV_WMI',
            'WISC_IV_PSI',
            'WISC_IV_SIM_SCALED',
            'WISC_IV_VOCAB_SCALED',
            'WISC_IV_INFO_SCALED',
            'WISC_IV_BLK_DSN_SCALED',
            'WISC_IV_PIC_CON_SCALED',
            'WISC_IV_MATRIX_SCALED',
            'WISC_IV_DIGIT_SPAN_SCALED',
            'WISC_IV_LET_NUM_SCALED',
            'WISC_IV_CODING_SCALED',
            'WISC_IV_SYM_SCALED',
            'EYE_STATUS_AT_SCAN',
            'AGE_AT_MPRAGE',
            'BMI',
           ]

for f in features:
    raw = Reader.get_subject_score(subject_IDs, score=f)
    temp = pd.DataFrame.from_dict(raw,orient='index',columns=[f])
    df = df.join(temp)
df = df.reset_index()
print('Done Retrieving ABIDE data.')

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# CLEAN
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
print('Cleaning Data...')
df["SEX"] = df["SEX"].astype(str).astype(int)
df["DX_GROUP"] = df["DX_GROUP"].astype(str).astype(int)

df["FIQ"] = normalizeMinMax(pd.to_numeric(df['FIQ']),30,170)
df["PIQ"] = normalizeMinMax(pd.to_numeric(df['PIQ']),31,169)
df["VIQ"] = normalizeMinMax(pd.to_numeric(df['VIQ']),31,166)


one_hot = pd.get_dummies(df['FIQ_TEST_TYPE'],prefix='FIQ__')
df = df.drop('FIQ_TEST_TYPE',axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['VIQ_TEST_TYPE'],prefix='VIQ__')
df = df.drop('VIQ_TEST_TYPE',axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['PIQ_TEST_TYPE'],prefix='PIQ__')
df = df.drop('PIQ_TEST_TYPE',axis = 1)
df = df.join(one_hot)

df["AGE_AT_SCAN"] = normalizeMinMax(pd.to_numeric(df['AGE_AT_SCAN']),6,64)
df["ADI_R_SOCIAL_TOTAL_A"] = normalizeMinMax(pd.to_numeric(df['ADI_R_SOCIAL_TOTAL_A']),0,30,shift=1)
df["ADI_R_VERBAL_TOTAL_BV"] = normalizeMinMax(pd.to_numeric(df['ADI_R_VERBAL_TOTAL_BV']),0,26,shift=1)
df["ADI_RRB_TOTAL_C"] = normalizeMinMax(pd.to_numeric(df['ADI_RRB_TOTAL_C']),0,12,shift=1)
df["ADI_R_ONSET_TOTAL_D"] = normalizeMinMax(pd.to_numeric(df['ADI_R_ONSET_TOTAL_D']),0,5,shift=1)
df["ADI_R_RSRCH_RELIABLE"] = normalizeMinMax(pd.to_numeric(df['ADI_R_RSRCH_RELIABLE']),0,1,shift=1)

df["ADOS_MODULE"] = normalizeMinMax(pd.to_numeric(df['ADOS_MODULE']),1,4)
df["ADOS_TOTAL"] = normalizeMinMax(pd.to_numeric(df['ADOS_TOTAL']),0,22,shift=1)
df["ADOS_COMM"] = normalizeMinMax(pd.to_numeric(df['ADOS_COMM']),0,8,shift=1)
df["ADOS_SOCIAL"] = normalizeMinMax(pd.to_numeric(df['ADOS_SOCIAL']),0,14,shift=1)
df["ADOS_STEREO_BEHAV"] = normalizeMinMax(pd.to_numeric(df['ADOS_STEREO_BEHAV']),0,8,shift=1)
df["ADOS_RSRCH_RELIABLE"] = normalizeMinMax(pd.to_numeric(df['ADOS_RSRCH_RELIABLE']),0,1,shift=1)
df["ADOS_GOTHAM_SOCAFFECT"] = normalizeMinMax(pd.to_numeric(df['ADOS_GOTHAM_SOCAFFECT']),0,20,shift=1)
df["ADOS_GOTHAM_RRB"] = normalizeMinMax(pd.to_numeric(df['ADOS_GOTHAM_RRB']),0,8,shift=1)
df["ADOS_GOTHAM_TOTAL"] = normalizeMinMax(pd.to_numeric(df['ADOS_GOTHAM_TOTAL']),0,28,shift=1)
df["ADOS_GOTHAM_SEVERITY"] = normalizeMinMax(pd.to_numeric(df['ADOS_GOTHAM_SEVERITY']),1,10)

df["SRS_VERSION"] = normalizeMinMax(pd.to_numeric(df['SRS_VERSION']),1,2)
df["SRS_RAW_TOTAL"] = normalizeMinMax(pd.to_numeric(df['SRS_RAW_TOTAL']),0,117,shift=1)
df["SRS_AWARENESS"] = normalizeMinMax(pd.to_numeric(df['SRS_AWARENESS']),0,19,shift=1)
df["SRS_COGNITION"] = normalizeMinMax(pd.to_numeric(df['SRS_COGNITION']),0,24,shift=1)
df["SRS_COMMUNICATION"] = normalizeMinMax(pd.to_numeric(df['SRS_COMMUNICATION']),0,43,shift=1)
df["SRS_MOTIVATION"] = normalizeMinMax(pd.to_numeric(df['SRS_MOTIVATION']),0,22,shift=1)
df["SRS_MANNERISMS"] = normalizeMinMax(pd.to_numeric(df['SRS_MANNERISMS']),0,22,shift=1)

df["SCQ_TOTAL"] = normalizeMinMax(pd.to_numeric(df['SCQ_TOTAL']),0,39,shift=1)
df["AQ_TOTAL"] = normalizeMinMax(pd.to_numeric(df['AQ_TOTAL']),0,50,shift=1)
# df["CURRENT_MED_STATUS"] = normalizeMinMax(pd.to_numeric(df['CURRENT_MED_STATUS']),0,1,shift=1)
df["OFF_STIMULANTS_AT_SCAN"] = normalizeMinMax(pd.to_numeric(df['OFF_STIMULANTS_AT_SCAN']),0,1,shift=1)

df["VINELAND_RECEPTIVE_V_SCALED"] = normalizeMinMax(pd.to_numeric(df['VINELAND_RECEPTIVE_V_SCALED']),1,24)
df["VINELAND_EXPRESSIVE_V_SCALED"] = normalizeMinMax(pd.to_numeric(df['VINELAND_EXPRESSIVE_V_SCALED']),1,24)
df["VINELAND_WRITTEN_V_SCALED"] = normalizeMinMax(pd.to_numeric(df['VINELAND_WRITTEN_V_SCALED']),1,24)
df["VINELAND_COMMUNICATION_STANDARD"] = normalizeMinMax(pd.to_numeric(df['VINELAND_COMMUNICATION_STANDARD']),20,160)
df["VINELAND_PERSONAL_V_SCALED"] = normalizeMinMax(pd.to_numeric(df['VINELAND_PERSONAL_V_SCALED']),1,24)
df["VINELAND_DOMESTIC_V_SCALED"] = normalizeMinMax(pd.to_numeric(df['VINELAND_DOMESTIC_V_SCALED']),1,24)
df["VINELAND_COMMUNITY_V_SCALED"] = normalizeMinMax(pd.to_numeric(df['VINELAND_COMMUNITY_V_SCALED']),1,24)
df["VINELAND_DAILYLVNG_STANDARD"] = normalizeMinMax(pd.to_numeric(df['VINELAND_DAILYLVNG_STANDARD']),20,160)
df["VINELAND_INTERPERSONAL_V_SCALED"] = normalizeMinMax(pd.to_numeric(df['VINELAND_INTERPERSONAL_V_SCALED']),1,24)
df["VINELAND_PLAY_V_SCALED"] = normalizeMinMax(pd.to_numeric(df['VINELAND_PLAY_V_SCALED']),1,24)
df["VINELAND_COPING_V_SCALED"] = normalizeMinMax(pd.to_numeric(df['VINELAND_COPING_V_SCALED']),1,24)
df["VINELAND_SOCIAL_STANDARD"] = normalizeMinMax(pd.to_numeric(df['VINELAND_SOCIAL_STANDARD']),20,160)
df["VINELAND_SUM_SCORES"] = normalizeMinMax(pd.to_numeric(df['VINELAND_SUM_SCORES']),76,480)
df["VINELAND_ABC_STANDARD"] = normalizeMinMax(pd.to_numeric(df['VINELAND_ABC_STANDARD']),20,160)
df["VINELAND_INFORMANT"] = normalizeMinMax(pd.to_numeric(df['VINELAND_INFORMANT']),1,2)

df["WISC_IV_VCI"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_VCI']),3,57)
df["WISC_IV_PRI"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_PRI']),3,57)
df["WISC_IV_WMI"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_WMI']),2,38)
df["WISC_IV_PSI"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_PSI']),2,38)
df["WISC_IV_SIM_SCALED"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_SIM_SCALED']),1,19)
df["WISC_IV_VOCAB_SCALED"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_VOCAB_SCALED']),1,19)
df["WISC_IV_INFO_SCALED"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_INFO_SCALED']),1,19)
df["WISC_IV_BLK_DSN_SCALED"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_BLK_DSN_SCALED']),1,19)
df["WISC_IV_PIC_CON_SCALED"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_PIC_CON_SCALED']),1,19)
df["WISC_IV_MATRIX_SCALED"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_MATRIX_SCALED']),1,19)
df["WISC_IV_DIGIT_SPAN_SCALED"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_DIGIT_SPAN_SCALED']),1,19)
df["WISC_IV_LET_NUM_SCALED"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_LET_NUM_SCALED']),1,19)
df["WISC_IV_CODING_SCALED"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_CODING_SCALED']),1,19)
df["WISC_IV_SYM_SCALED"] = normalizeMinMax(pd.to_numeric(df['WISC_IV_SYM_SCALED']),1,19)

df["EYE_STATUS_AT_SCAN"] = normalizeMinMax(pd.to_numeric(df['EYE_STATUS_AT_SCAN']),1,2)
df["AGE_AT_MPRAGE"] = normalizeMinMax(pd.to_numeric(df['AGE_AT_MPRAGE']),6,64)
df["BMI"] = normalizeMinMax(pd.to_numeric(df['BMI']),15,40)

df['site_id'] = pd.factorize(df['SITE_ID'])[0] # factorize site


df["is_male"] = (df["SEX"]==1).astype(int)
df["is_autism"] = (df["DX_GROUP"]==1).astype(int)

site_key = df[['site_id','SITE_ID']]

df = df.drop(columns=['SEX','DX_GROUP'])
df = df.drop(columns=['SITE_ID'])
df = df.rename({'index': 'subject_id'},
            axis='columns')


site_key = site_key.groupby(['site_id','SITE_ID']).size().reset_index()
print('Site key: ')
print(site_key)

print('Features with label correlation > 0.55: ')
corr = df.corr()
print(corr[corr['is_autism'].abs() > 0.55]['is_autism'])

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# CHECK SITES
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
print('Checking sites...')
site_summary = get_site_summary(df)
grouped = split_groups(site_summary,sex_boundary = SEX_BOUNDARY)
df = df.join(grouped[['site_id','group']].set_index('site_id'),on='site_id')
print(grouped.sort_values(by=['group','site_total'],ascending=False))


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# PREPARE NODE INFO FOR EACH GROUP
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
print('Preparing node info....')
# group 1
labels_g1 = df[df['group'] == 1]['is_autism'].to_numpy()
has_label_g1 = np.ones(labels_g1.shape,dtype=bool)
nodes_g1 = df[df['group'] == 1]

print('Checking Group 1 corr')
corr = nodes_g1.corr()
print(corr[corr['is_autism'].abs() > 0.55]['is_autism'])

# group 2
labels_g2 = df[df['group'] == 2]['is_autism'].to_numpy()
has_label_g2 = np.ones(labels_g2.shape,dtype=bool)
nodes_g2 = df[df['group'] == 2]
print(nodes_g2.shape)

print('Checking Group 2 corr')
corr = nodes_g2.corr()
print(corr[corr['is_autism'].abs() > 0.55]['is_autism'])


corr_cols = ['ADOS_RSRCH_RELIABLE','ADOS_MODULE','ADI_R_RSRCH_RELIABLE'] # mod 1
corr_cols = corr_cols + ['ADI_R_SOCIAL_TOTAL_A', 'ADI_R_VERBAL_TOTAL_BV']
corr_cols = corr_cols + ['ADI_RRB_TOTAL_C','ADI_R_ONSET_TOTAL_D','ADOS_GOTHAM_SEVERITY']
# corr_cols = []
remove_cols = ['subject_id','group','is_autism','is_male'] + corr_cols
# NOTE: change here if prot att included in node feat
nodes_g1 = nodes_g1.drop(columns=remove_cols).to_numpy()
# nodes_g1 = nodes_g1.drop(columns=['subject_id','group','is_autism','is_male']).to_numpy()
nodes_g1 = nodes_g1.astype(float)
prot_g1 = df[df['group'] == 1]['is_male'].to_numpy()

nodes_g2 = nodes_g2.drop(columns=remove_cols).to_numpy()
# nodes_g2 = nodes_g2.drop(columns=['subject_id','group','is_autism','is_male']).to_numpy()
nodes_g2 = nodes_g2.astype(float)
prot_g2 = df[df['group'] == 2]['is_male'].to_numpy()


# prepare multiclass labels
label_g1_multi = np.ones((labels_g1.shape[0],2),dtype=bool)
label_g1_multi[:,1]=labels_g1
label_g1_multi[:,0]=np.abs(labels_g1-1)

label_g2_multi = np.ones((labels_g2.shape[0],2),dtype=bool)
label_g2_multi[:,1]=labels_g2
label_g2_multi[:,0]=np.abs(labels_g2-1)


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# DISPLAY A GROUP SUMMARY
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
print('Printing group summary...')
summary = grouped.groupby(['group'])[['site_total','site_male','site_autism']].sum().reset_index()
summary['p_male'] = (summary['site_male']/summary['site_total']).round(2)
summary['p_autism'] = (summary['site_autism']/summary['site_total']).round(2)
print(summary)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# CONSTRUCT SEPARATE ADJACENCY MATRICIES
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
print('Constructing the adjacency matrices...')
img_data = Reader.get_networks(subject_IDs, kind='correlation', atlas_name='ho')
img_df = pd.DataFrame(img_data)

# group 1
i_g1 = np.where(df['group'] == 1)[0]
img_g1 = np.take(img_data,i_g1,axis=0)
adj_g1 = getAdjacencyMatrix(img_g1,SIMILARITY_THRESHOLD)
# adj_g1 = sparse.csr_matrix(adj_g1)

# group 2
i_g2 = np.where(df['group'] == 2)[0]
img_g2 = np.take(img_data,i_g2,axis=0)
adj_g2 = getAdjacencyMatrix(img_g2,SIMILARITY_THRESHOLD)
# adj_g2 = sparse.csr_matrix(adj_g2)

print('Source Graph Info'+getGraphInfoString(nodes_g1,has_label_g1,labels_g1,prot_g1,adj_g1))
print('Target Graph Info'+getGraphInfoString(nodes_g2,has_label_g2,labels_g2,prot_g2,adj_g2))

adj_g1 = sparse.csr_matrix(adj_g1)
adj_g2 = sparse.csr_matrix(adj_g2)


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# SAVING DATA
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
OUTPUT_SOURCE_FILEPATH = 'data/abide_large.mat'
print(f'Saving source data to: {OUTPUT_SOURCE_FILEPATH}')
mData_source = {}
mData_source['network'] = adj_g1
mData_source['attrb'] = nodes_g1
mData_source['prot'] = prot_g1
mData_source['group'] = label_g1_multi
scipy.io.savemat(OUTPUT_SOURCE_FILEPATH, mData_source)

OUTPUT_TARGET_FILEPATH = 'data/abide_small.mat'
print(f'Saving target data to: {OUTPUT_TARGET_FILEPATH}')
mData_target = {}
mData_target['network'] = adj_g2
mData_target['attrb'] = nodes_g2
mData_target['prot'] = prot_g2
mData_target['group'] = label_g2_multi
scipy.io.savemat(OUTPUT_TARGET_FILEPATH, mData_target)