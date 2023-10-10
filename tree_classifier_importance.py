import numpy as np
import pandas as pd
from pandas import read_csv
from skimage.measure import fit
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split

def get_pca():
    # load data
    filename = 'breast-cancer-wisconsin.csv'
    cancer = pd.read_csv('breast-cancer-wisconsin-Copy.csv')
    feature_names = list(cancer.keys())
    dataframe = read_csv(filename, names=feature_names)
    # load data
    array = dataframe.values
    x = array[:, 0:30]
    y = array[:, 30]

    # feature extraction
    pca = PCA(n_components=20)
    x_scaled = StandardScaler().fit_transform(x)

    # Fit and transform data
    pca_features = pca.fit_transform(x_scaled)

    # Create dataframe
    pca_df = pd.DataFrame(
        data=pca_features,
        columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14',
                 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20'])
    # map target names to PCA features
    target_names = {
        0: 'no',
        1: 'yes',
    }

    pca_df['target'] = y
    pca_df['target'] = pca_df['target'].map(target_names)
    print(pca_df.head(100))
    # summarize components
    # print("Explained Variance: %s" % fit.explained_variance_ratio_)
    # print(fit.components_)
    pca_df.to_csv('pca_result.csv', header=True, index=False)
    return pca_df

# --------------------------------------

def get_anova():
    # loading the dataset
    filename = 'breast-cancer-wisconsin.csv'
    cancer = pd.read_csv('breast-cancer-wisconsin-Copy.csv')
    feature_names = list(cancer.keys())
    dataframe = read_csv(filename, names=feature_names)
    # load data
    array = dataframe.values
    x = array[:, 0:29]
    y = array[:, 29]
    # feature extraction
    test = SelectKBest(score_func=f_classif, k=20)
    fit = test.fit(x, y)
    # summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(x)
    # summarize selected features
    # print(features[0:, :])

    features["diagnosis"] = y[:]
    # Fit and transform data
    # Create dataframe
    anova_df = pd.DataFrame(data=features)
    print(anova_df.head(10))
    anova_df.to_csv('anova_result.csv', index=False)
    return anova_df

# --------------------------------------
# def get_importance(ds, x, y):
def get_random_forest():
    output = []
    # loading the dataset
    trans_data = pd.read_csv('breast-cancer-wisconsin-Copy.csv').to_numpy()
    x_data, y_data = trans_data[:, 0:-1], trans_data[:, -1]
    df = pd.read_csv('breast-cancer-wisconsin-Copy.csv')
    feature_names = list(df.keys())
    print("-----------------------------")
    # +++++++++++++++++++++++++++++++++++++++++++++++
    feat_labels = df.columns[1:]
    forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=1)
    forest.fit(x_data, y_data)
    sfm = SelectFromModel(forest, threshold=0.001)
    sfm.fit(x_data, y_data)
    # Print the names of the most important features
    i = 0
    """for feature_list_index in sfm.get_support(indices=True):
        print("i", i, "feature", feat_labels[feature_list_index])
        i += 1"""
    o = []
    importance = forest.feature_importances_
    indices = np.argsort(importance)[::-1]
    j = 0
    for f in range(x_data.shape[1]):
        if j < 20:
            # print(f, feat_labels[f], importance[indices[f]])
            output.append([feat_labels[f], importance[indices[f]]])
            o.append(feat_labels[f])
            j += 1

    d = df.loc[:, o]
    d["diagnosis"] = y_data[:]
    print(d.head(20))
    d.to_csv("tree_result.csv", header=True, index=False)
    return o

    # Fit and transform data
    # Create dataframe

# --------------------------------------------------

get_random_forest()
# get_anova()
# print(o)
# get_pca()
