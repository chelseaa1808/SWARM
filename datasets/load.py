from os.path import dirname, join
import pandas as pd
import numpy as np

def load_expression_dataset(folder: str, ctl_file: str, ad_file: str, label_ctl="CTL", label_ad="AD") -> tuple:
    """
    Loads and processes gene expression data from control and AD files.

    Returns:
        X (np.ndarray): Expression matrix
        y (np.ndarray): Encoded labels
        probes (np.ndarray): Feature/gene names
        classes (np.ndarray): Label classes
        folder (str): Folder path
    """
    with open(join(folder, "common_probes.txt"), "r") as fr:
        probes = [p for p in fr.read().splitlines() if p]
    probes = np.array(probes)

    df_ctl = pd.read_csv(join(folder, ctl_file), sep='\t', index_col=0).T
    df_ctl['label'] = label_ctl
    df_ad = pd.read_csv(join(folder, ad_file), sep='\t', index_col=0).T
    df_ad['label'] = label_ad

    df = pd.concat([df_ad, df_ctl], axis=0, join='inner', verify_integrity=True)
    labels = df.pop('label')
    df = df.reindex(probes, axis='columns')
    df.insert(0, 'label', labels)
    df = df.sample(frac=1, random_state=1)

    classes = df.label.unique()
    X = df.iloc[:, 1:].values.astype(float)
    y = df.label.astype('category').cat.codes.values
    return X, y, probes, classes, folder

def load_ad_blood1(): return load_expression_dataset(join(dirname(__file__), 'ad_blood1'), "blood1_ctl.txt", "blood1_ad.txt")
def load_ad_blood2(): return load_expression_dataset(join(dirname(__file__), 'ad_blood2'), "blood2_ctl.txt", "blood2_ad.txt")
def load_ad_kronos(): return load_expression_dataset(join(dirname(__file__), 'ad_kronos'), "kronos_ctl.txt", "kronos_ad.txt")
def load_ad_rush(): return load_expression_dataset(join(dirname(__file__), 'ad_rush'), "rush_ctl.txt", "rush_ad.txt")



# def load_ad_somalogic():
#     # set the folders
#     #folder = join(dirname(__file__), '..', 'datasets', 'ad_somalogic')
#     folder = join(dirname(__file__), 'ad_somalogic')
#
#     df_ctl = pd.read_csv(join(folder, "Somalogic_control.txt"), index_col=0, sep='\t')
#     genes = df_ctl.index.values
#     df_ctl = df_ctl.T;
#     df_ctl['label'] = 'CTL'
#     df_ad = pd.read_csv(join(folder, "Somalogic_LOAD.txt"), index_col=0, sep='\t')
#     df_ad = df_ad.T;
#     df_ad['label'] = 'AD'
#     df = pd.concat([df_ad, df_ctl], axis=0, join='inner', verify_integrity=True)
#     df_label = df.label;
#     df.pop('label');
#     df.insert(0, 'label', df_label)
#     df = df.sample(frac=1, replace=True, random_state=1)
#
#     classes = df.label.unique()
#     X = df.iloc[:, 1:].values.astype(float)
#     y = df.label.astype('category').cat.codes.values
#
#     return X, y, genes, classes, folder
#
#
# def load_leukemia2():
#
#     #folder = join(dirname(__file__), '..', 'datasets', 'leukemia2')
#     folder = join(dirname(__file__), 'leukemia2')
#     df = pd.read_csv(join(folder, 'leukemia2.txt'), index_col=0, sep='\t')
#     df.pop('Description')
#     df = df.T
#     df['label'] = df.index.str[:3]
#     label = df.label
#     df.pop('label')
#     df.insert(0, 'label', label)
#     df = df.sample(frac=1, replace=True)
#     genes = df.columns[1:].values
#     classes = df.label.unique()
#     X = df.iloc[:,1:].values.astype(float)
#     X = (X - X.min()) / (X.max() - X.min())
#     y = df.label.astype('category').cat.codes.values
#
#     return X, y, genes, classes, folder
#
#
# def load_prostate_tumor():
#     #folder = join(dirname(__file__), '..', 'datasets', 'prostate_tumor')
#     folder = join(dirname(__file__), 'prostate_tumor')
#     df = pd.read_csv(join(folder, 'prostate_tumor.csv'), index_col=0)
#     df = df.rename(index=str, columns={'class': 'label'})
#     genes = pd.read_csv(join(folder, 'prostate_gene.txt'), index_col=0, sep='\t', header=None)
#     genes.index = genes.index.map(str)
#     df1 = df.iloc[:, 0:1]
#     df2 = df.T.iloc[1:, :]
#     df3 = genes.merge(df2, left_index=True, right_index=True, how='inner')
#     df3 = df3.rename(index=str, columns={1: "index"})
#     df3 = df3.reset_index().set_index('index').drop(columns=[ 'level_0' ], axis=1)
#     df = df1.merge(df3.T, left_index=True, right_index=True, how='inner')
#     df = df.sample(frac=1, replace=True)
#     genes = df.columns[1: ].values
#     classes = df.label.unique()
#     X = df.iloc[:, 1:].values.astype(float)
#     # X = (X - X.min())/(X.max() - X.min())
#     y = df.label.astype('category').cat.codes.values
#
#     return X, y, genes, classes, folder
#
# def load_dlbcl():
#     #folder = join(dirname(__file__), '..', 'datasets', 'dlbcl')
#     folder = join(dirname(__file__), 'dlbcl')
#     df = pd.read_csv(join(folder, 'dlbcl.csv'), index_col=0)
#     df = df.rename(index=str, columns={'class': 'label'})
#     df.label.replace('Diffuse large B-cell Lymphoma', 'DLBCL', inplace=True)
#     df.label.replace('Follicular Lymphoma', 'FL', inplace=True)
#     genes = pd.read_csv(join(folder, 'dlbcl_gene.txt'), index_col=0, sep=' ')
#     genes.index = genes.index.map(str)
#     df1 = df.iloc[:,0:1]
#     df2 = df.T.iloc[1:,:]
#     df3 = genes.merge(df2, left_index=True, right_index=True, how='inner')
#     df3 = df3.rename(index=str, columns={1: "index"})
#     df3 = df3.reset_index().set_index('Gene_Symbol').drop(columns=['index'], axis=1)
#     df = df1.merge(df3.T, left_index=True, right_index=True, how='inner')
#     df = df.sample(frac=1, replace=True)
#     genes = df.columns[1:].values
#     classes = df.label.unique()
#     X = df.iloc[:,1:].values.astype(float)
#     X = (X - X.min())/(X.max() - X.min())
#     y = df.label.astype('category').cat.codes.values
#
#     return X, y, genes, classes, folder
#
# def load_ptsd1():
#
#     folder = join(dirname(__file__), 'ptsd1')
#     df_dis = pd.read_csv(join(folder, 'PTSD_GE_MODEL1_PSS_disease_HGNC.txt'), index_col=0, sep=' ')
#     df_dis = df_dis.reset_index().dropna().drop_duplicates('geneid').set_index('geneid')
#     df_ctl = pd.read_csv(join(folder, 'PTSD_GE_MODEL1_PSS_control_HGNC.txt'), index_col=0, sep=' ')
#     df_ctl = df_ctl.reset_index().dropna().drop_duplicates('geneid').set_index('geneid')
#     df = df_dis.merge(df_ctl, left_index=True, right_index=True, how='inner')
#     df = df.reindex(sorted(df.columns), axis=1)
#
#     y = pd.read_csv(join(folder, 'PTSD_COV_MODEL1_PSS.txt'), sep='\t', index_col=0)
#     df = y.T.merge(df.T, left_index=True, right_index=True, how='inner')
#     df = df.sample(frac=1, replace=True)
#
#     genes = df.columns[1:]
#     classes = df.PSS.unique()
#     X = df.iloc[:, 1:].values.astype(float)
#     y = df.PSS.values.astype(int)
#
#     return X, y, genes, classes, folder
#
# def load_ptsd2():
#
#     folder = join(dirname(__file__), 'ptsd2')
#     df_dis = pd.read_csv(join(folder, 'PTSD_GE_MODEL2_BDI_disease_HGNC.txt'), index_col=0, sep=' ')
#     df_dis = df_dis.reset_index().dropna().drop_duplicates('geneid').set_index('geneid')
#     df_ctl = pd.read_csv(join(folder, 'PTSD_GE_MODEL2_BDI_control_HGNC.txt'), index_col=0, sep=' ')
#     df_ctl = df_ctl.reset_index().dropna().drop_duplicates('geneid').set_index('geneid')
#     df = df_dis.merge(df_ctl, left_index=True, right_index=True, how='inner')
#     df = df.reindex(sorted(df.columns), axis=1)
#
#     y = pd.read_csv(join(folder, 'PTSD_COV_MODEL2_BDI.txt'), sep='\t', index_col=0)
#     df = y.T.merge(df.T, left_index=True, right_index=True, how='inner')
#     df = df.sample(frac=1, replace=True)
#
#     genes = df.columns[1:]
#     classes = df.BDI.unique()
#     X = df.iloc[:, 1:].values.astype(float)
#     y = df.BDI.values.astype(int)
#
#     return X, y, genes, classes, folder
#
# def load_ptsd3():
#
#     folder = join(dirname(__file__), 'ptsd3')
#     df_dis = pd.read_csv(join(folder, 'PTSD_GE_MODEL3_PSSXBDI_disease_HGNC.txt'), index_col=0, sep=' ')
#     df_dis = df_dis.reset_index().dropna().drop_duplicates('geneid').set_index('geneid')
#     df_ctl = pd.read_csv(join(folder, 'PTSD_GE_MODEL3_PSSXBDI_control_HGNC.txt'), index_col=0, sep=' ')
#     df_ctl = df_ctl.reset_index().dropna().drop_duplicates('geneid').set_index('geneid')
#     df = df_dis.merge(df_ctl, left_index=True, right_index=True, how='inner')
#     df = df.reindex(sorted(df.columns), axis=1)
#
#     y = pd.read_csv(join(folder, 'PTSD_COV_MODEL3_PSSXBDI.txt'), sep='\t', index_col=0)
#     df = y.T.merge(df.T, left_index=True, right_index=True, how='inner')
#     df.rename(index=str, columns={'PSSXBDI 0': 'PSSXBDI'}, inplace=True)
#     df = df.sample(frac=1, replace=True)
#
#     genes = df.columns[1:].values
#     classes = df.PSSXBDI.unique()
#     X = df.iloc[:, 1:].values.astype(float)
#     y = df.PSSXBDI.values.astype(int)
#
#     return X, y, genes, classes, folder
#
# def load_synthetic1():
#     folder = join(dirname(__file__), 'synthetic')
#     n_samples = 200
#     n_features = 10
#     genes = np.arange(n_features)
#     X = np.random.rand(n_samples, n_features)
#     X[:, 2], X[:, 3] = X[:, 0], X[:, 1]
#     y = np.array([1 if (X[i, 0] + X[i, 1]) / 2 > 0.5 else 0 for i in range(n_samples)])
#     return X, y, genes, None, folder
#
#
# def load_synthetic2():
#     folder = join(dirname(__file__), 'synthetic')
#     n_samples = 200
#     n_features = 10000
#     genes = np.arange(n_features)
#     X = np.random.rand(n_samples, n_features)
#     X[:,2] = (X[:,0]+X[:,1])/2
#     X[:,3] = X[:,0]
#     y = np.array([1 if X[i, 2] > 0.5 else 0 for i in range(n_samples)])
#
#     return X, y, genes, None, folder
#
# def load_monks():
#     folder = join(dirname(__file__), '..', 'datasets', 'synthetic')
#
#     df = pd.read_csv(join(folder, 'monks-3.txt'), header=None, sep=' ')
#     n_features = 10000
#     genes = np.arange(n_features)
#     # 6 discrete features with best subset being {1,3,4}
#     X = df.iloc[:, 2:8].values.astype(int)
#     # append random noise to reach n_features
#     X = np.append(X, np.random.rand(X.shape[0], n_features-X.shape[1]), axis=1)
#     y = df[1].values.astype(int)
#
#     return X, y, genes, None, folder
#
#
# def load_imp():
#     #folder = join(dirname(__file__), '..', 'datasets', 'imp')
#     folder = join(dirname(__file__), 'imp')
#
#     df = pd.read_csv(join(folder, "RTT_SN_BON.csv"), header=0, index_col=0)
#     genes = df.columns.values
#     X = df.values.astype(float)
#
#     return X, None, genes, None, folder
#
# def load_colon():
#     folder = join(dirname(__file__), '..', 'datasets', 'Cancer_Data', 'COLON')
#     # folder = join(dirname(__file__), 'COLON')
#
#     df_c = pd.read_csv(join(folder, "COLON_GTEX_control.txt"), index_col=0, sep='\t')
#     df_c.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
#     genes = df_c.index.values.astype(np.str)
#     genes.sort()
#     df_c = df_c.T
#     df_c['label'] = 'C'
#     df_d = pd.read_csv(join(folder, "COLON_TCGA_disease.txt"), index_col=0, sep='\t')
#     df_d.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
#     df_d = df_d.T
#     df_d['label'] = 'D'
#     df = pd.concat([df_d, df_c], axis=0, join='outer', verify_integrity=True)
#     df_label = df.label
#     df.pop('label')
#     df.sort_index(axis=1, inplace=True)
#     df.insert(0, 'label', df_label)
#     df = df.sample(frac=1, replace=True, random_state=1)
#     classes = df.label.unique()
#     X = df.iloc[:, 1:].values.astype(float)
#     y = df.label.astype('category').cat.codes.values
#
#     return X, y, genes, classes, folder
#
#
# def load_breast():
#     folder = join(dirname(__file__), '..', 'datasets', 'Cancer_Data', 'BREAST')
#     # folder = join(dirname(__file__), 'BREAST')
#
#     df_c = pd.read_csv(join(folder, "BRCA_GTEX_control.txt"), index_col=0, sep='\t')
#     df_c.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
#     genes = df_c.index.values.astype(np.str)
#     genes.sort()
#     df_c = df_c.T
#     df_c['label'] = 'C'
#     df_d = pd.read_csv(join(folder, "BRCA_TCGA_disease.txt"), index_col=0, sep='\t')
#     df_d.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
#     df_d = df_d.T
#     df_d['label'] = 'D'
#     df = pd.concat([df_d, df_c], axis=0, join='outer', verify_integrity=True)
#     df_label = df.label
#     df.pop('label')
#     df.sort_index(axis=1, inplace=True)
#     df.insert(0, 'label', df_label)
#     df = df.sample(frac=1, replace=True, random_state=1)
#     classes = df.label.unique()
#     X = df.iloc[:, 1:].values.astype(float)
#     y = df.label.astype('category').cat.codes.values
#
#     return X, y, genes, classes, folder
#
# def load_prostate():
#     folder = join(dirname(__file__), '..', 'datasets', 'Cancer_Data', 'PROSTATE')
#     # folder = join(dirname(__file__), 'PROSTATE')
#
#     df_c = pd.read_csv(join(folder, "PROSTATE_GTEX_control.txt"), index_col=0, sep='\t')
#     df_c.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
#     genes = df_c.index.values.astype(np.str)
#     genes.sort()
#     df_c = df_c.T
#     df_c['label'] = 'C'
#     df_d = pd.read_csv(join(folder, "PROSTATE_TCGA_disease.txt"), index_col=0, sep='\t')
#     df_d.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
#     df_d = df_d.T
#     df_d['label'] = 'D'
#     df = pd.concat([df_d, df_c], axis=0, join='outer', verify_integrity=True)
#     df_label = df.label
#     df.pop('label')
#     df.sort_index(axis=1, inplace=True)
#     df.insert(0, 'label', df_label)
#     df = df.sample(frac=1, replace=True, random_state=1)
#     classes = df.label.unique()
#     X = df.iloc[:, 1:].values.astype(float)
#     y = df.label.astype('category').cat.codes.values
#
#     return X, y, genes, classes, folder
#
#
# def load_liver():
#     folder = join(dirname(__file__), '..', 'datasets', 'LIVER')
#     # folder = join(dirname(__file__), 'LIVER')
#
#     df_c = pd.read_csv(join(folder, "LIVER_GTEX_control.txt"), index_col=0, sep='\t')
#     df_c.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
#     genes = df_c.index.values.astype(np.str)
#     genes.sort()
#     df_c = df_c.T
#     df_c['label'] = 'C'
#     df_d = pd.read_csv(join(folder, "LIVER_TCGA_disease.txt"), index_col=0, sep='\t')
#     df_d.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
#     df_d = df_d.T
#     df_d['label'] = 'D'
#     df = pd.concat([df_d, df_c], axis=0, join='outer', verify_integrity=True)
#     df_label = df.label
#     df.pop('label')
#     df.sort_index(axis=1, inplace=True)
#     df.insert(0, 'label', df_label)
#     df = df.sample(frac=1, replace=True, random_state=1)
#     classes = df.label.unique()
#     X = df.iloc[:, 1:].values.astype(float)
#     y = df.label.astype('category').cat.codes.values
#
#     return X, y, genes, classes, folder
#
#
# def load_thyroid():
#     folder = join(dirname(__file__), '..', 'datasets',  'THYROID')
#     # folder = join(dirname(__file__), 'LANG')
#
#     df_c = pd.read_csv(join(folder, "THYROID_GTEX_control.txt"), index_col=0, sep='\t')
#     df_c.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
#     genes = df_c.index.values.astype(np.str)
#     genes.sort()
#     df_c = df_c.T
#     df_c['label'] = 'C'
#     df_d = pd.read_csv(join(folder, "THYROID_TCGA_disease.txt"), index_col=0, sep='\t')
#     df_d.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
#     df_d = df_d.T
#     df_d['label'] = 'D'
#     df = pd.concat([df_d, df_c], axis=0, join='outer', verify_integrity=True)
#     df_label = df.label
#     df.pop('label')
#     df.sort_index(axis=1, inplace=True)
#     df.insert(0, 'label', df_label)
#     df = df.sample(frac=1, replace=True, random_state=1)
#     classes = df.label.unique()
#     X = df.iloc[:, 1:].values.astype(float)
#     y = df.label.astype('category').cat.codes.values
#
#     return X, y, genes, classes, folder
#
#
# def load_stomach():
#     folder = join(dirname(__file__), '..', 'datasets', 'STOMACH')
#     # folder = join(dirname(__file__), 'STOMACH')
#
#     df_c = pd.read_csv(join(folder, "STOMACH_GTEX_control.txt"), index_col=0, sep='\t')
#     df_c.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
#     genes = df_c.index.values.astype(np.str)
#     genes.sort()
#     df_c = df_c.T
#     df_c['label'] = 'C'
#     df_d = pd.read_csv(join(folder, "STOMACH_TCGA_disease.txt"), index_col=0, sep='\t')
#     df_d.drop(['Entrez_Gene_Id'], axis=1, inplace=True)
#     df_d = df_d.T
#     df_d['label'] = 'D'
#     df = pd.concat([df_d, df_c], axis=0, join='outer', verify_integrity=True)
#     df_label = df.label
#     df.pop('label')
#     df.sort_index(axis=1, inplace=True)
#     df.insert(0, 'label', df_label)
#     df = df.sample(frac=1, replace=True, random_state=1)
#     classes = df.label.unique()
#     X = df.iloc[:, 1:].values.astype(float)
#     y = df.label.astype('category').cat.codes.values
#
#     return X, y, genes, classes, folder

_loads = {
    'ad_blood1': load_ad_blood1,
    'ad_blood2': load_ad_blood2,
    'ad_kronos': load_ad_kronos,
    'ad_rush': load_ad_rush,
    # Future: 'colon': load_colon, ...
}
