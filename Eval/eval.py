#金标准
#mrna:disgenet2  (genecards1)    CGC1    OMIM1   NCG1
#lncrna:lnc2cancer1  lncrnadisease1   mndr2    lncACTdb1
#mirna:HMDD1   mndr2   mir2disease1    oncomirdb1   mircancer1   msdd1
import pandas as pd
DEL=pd.read_csv('../data/RAW_counts/PAAD_delncRNA_counts.csv')['Unnamed: 0'].tolist()
DEG=pd.read_csv('../data/RAW_counts/PAAD_demRNA_counts.csv')['Unnamed: 0'].tolist()
DEM=pd.read_csv('../data/RAW_counts/PAAD_demiRNA_counts.csv')['Unnamed: 0'].tolist()

#mrna:disgenet2  genecards1    CGC1    OMIM1   NCG1
#lncrna:lnc2cancer1  lncrnadisease1   mndr2    lncACTdb1
#mirna:HMDD1   mndr2   mi2disease1    oncomirdb1   mircancer1   msdd1

#lnc  input：lncrna list   output：association
def lncrna(rnalist):
    rele_counts=[]
    #lnc2cancer
    lnc2cancer=pd.read_excel('../data/eval_data/lnc2cancer.xlsx')[['name','cancer type']]
    pick=lnc2cancer.loc[lnc2cancer['cancer type'].str.contains('pan',case=False)]['name'].tolist()
    for l in rnalist:
        if l in set(pick):
            rele_counts+=[l]
            # print(l+' lnc2cancer')
    #lncrnadisease
    #旧
    # lncrnadisease=pd.read_excel('../data/eval_data/lncrnadisease_experimental lncRNA-disease information.xlsx')[['ncRNA Symbol','Disease Name']]
    # 新
    lncrnadisease=pd.read_excel('../data/eval_data_new/lncrnadisease_result3.xlsx')[['ncRNA_Symbol','Disease_Name']]
    # pick=lncrnadisease.loc[lncrnadisease['Disease Name'].str.contains('pan',case=False)]['ncRNA Symbol'].tolist()
    pick=lncrnadisease.loc[lncrnadisease['Disease_Name'].str.contains('pan',case=False)]['ncRNA_Symbol'].tolist()
    for l in rnalist:
        if l in set(pick):
            rele_counts += [l]
            # print(l + ' lncrnadisease')
    #mndr
    #旧
    # mndr=pd.read_excel('../data/eval_data/MNDR_Experimental lncRNA-disease information.xlsx')[['ncRNA Symbol','Disease']]
    #新
    mndr=pd.read_excel('../data/eval_data_new/RNADiseasev4.0_RNA-disease_experiment_all.xlsx')[['RNA Symbol','Disease Name']]
    # pick=mndr.loc[mndr['Disease'].str.contains('pan', case=False)]['ncRNA Symbol'].tolist()
    pick=mndr.loc[mndr['Disease Name'].str.contains('pan', case=False)]['RNA Symbol'].tolist()
    for l in rnalist:
        for p in set(pick):
            if l in p:
                rele_counts += [l]
                # print(l+' mndr')

    #lncACTdb
    #旧
    # lncACTdb=pd.read_excel('../data/eval_data/LncACTdb3.0Manual curation of experimentally validated cancer biomarkers.xlsx')[['biomarker name','cancer type']]
    #新
    lncACTdb=pd.read_excel('../data/eval_data_new/3.Manual curation of experimentally validated cancer biomarkers.xlsx')[['biomarker name','cancer type']]
    lncACTdb=lncACTdb.dropna(axis=0, subset=["cancer type"])
    pick=lncACTdb.loc[lncACTdb['cancer type'].str.contains('pan', case=False)]['biomarker name'].tolist()
    for l in rnalist:
        for p in set(pick):
            if l in p:
                rele_counts += [l]
                # print(l+' lncACTdb')

    # print(len(set(rele_counts)))
    # return len(set(rele_counts))
    return set(rele_counts)

def mrna(rnalist):
    rele_counts =[]
    #disgenet_all
    disgenet_all=pd.read_csv('../data/eval_data/disgenet_all_gene_disease_associations.tsv',delimiter='\t')[['geneSymbol','diseaseName']]
    pick=disgenet_all.loc[disgenet_all['diseaseName'].str.contains('pan', case=False)]['geneSymbol'].tolist()
    for l in rnalist:
        if l in set(pick):
            rele_counts += [l]
            # print(l + ' disgenet_all')

    #disgenet_curated
    disgenet_curated = pd.read_csv('../data/eval_data/disgenet_curated_gene_disease_associations.tsv', delimiter='\t')[['geneSymbol', 'diseaseName']]
    pick = disgenet_curated.loc[disgenet_curated['diseaseName'].str.contains('pan', case=False)]['geneSymbol'].tolist()
    for l in rnalist:
        if l in set(pick):
            rele_counts += [l]
            # print(l + ' disgenet_curated')

    #CGC
    pick = pd.read_csv('../data/eval_data/CGC_PAAD.csv')['Gene Symbol'].tolist()
    for l in rnalist:
        if l in set(pick):
            rele_counts += [l]
            # print(l + ' CGC')

    #OMIM
    pick = pd.read_excel('../data/eval_data/OMIM-Gene-Map-Retrieval_PAAD.xlsx',header=None)[0].tolist()
    for l in rnalist:
        if l in set(pick):
            rele_counts += [l]
            # print(l + ' OMIM')

    #NCG
    # 旧
    # NCG = pd.read_csv('../data/eval_data/NCG_cancerdrivers_annotation_supporting_evidence.tsv', delimiter='\t')[['symbol','cancer_type']]
    # 新
    NCG = pd.read_csv('../data/eval_data_new/NCG_cancerdrivers_annotation_supporting_evidence.tsv', delimiter='\t')[['symbol','cancer_type']]
    NCG = NCG.dropna(axis=0, subset=["cancer_type"])
    pick = NCG.loc[NCG['cancer_type'].str.contains('pan', case=False)]['symbol'].tolist()
    for l in rnalist:
        if l in set(pick):
            rele_counts += [l]
            # print(l + ' NCG')

    #mndr
    # 旧
    # mndr=pd.read_excel('../data/eval_data/MNDR_Experimental lncRNA-disease information.xlsx')[['ncRNA Symbol','Disease']]
    # 新
    mndr=pd.read_excel('../data/eval_data_new/RNADiseasev4.0_RNA-disease_experiment_all.xlsx')[['RNA Symbol','Disease Name']]
    # pick=mndr.loc[mndr['Disease'].str.contains('pan', case=False)]['ncRNA Symbol'].tolist()
    pick=mndr.loc[mndr['Disease Name'].str.contains('pan', case=False)]['RNA Symbol'].tolist()
    for l in rnalist:
        for p in set(pick):
            if l in p:
                rele_counts += [l]
                # print(l+' mndr')

    # print(len(set(rele_counts)))
    # return len(set(rele_counts))
    return set(rele_counts)


def mirna(rnalist):
    rele_counts = []
    #HMDD
    # 旧
    # hmdd = pd.read_excel('../data/eval_data/HMDD.xlsx')[['mir','disease']]
    # 新
    hmdd = pd.read_excel('../data/eval_data_new/HMDD.xlsx')[['miRNA','disease']]
    pick = hmdd.loc[hmdd['disease'].str.contains('pan', case=False)]['miRNA'].tolist()
    # pick = hmdd.loc[hmdd['disease'].str.contains('pan', case=False)]['mir'].tolist()
    for l in rnalist:
        if l in set(pick):
            rele_counts += [l]
            # print(l + ' hmdd')

    #mndr
    mndr = pd.read_excel('../data/eval_data/MNDR_Experimental miRNA-disease information.xlsx')[['ncRNA symbol','Disease']]
    pick = mndr.loc[mndr['Disease'].str.contains('pan', case=False)]['ncRNA symbol'].tolist()
    for l in rnalist:
        if l in set(pick):
            rele_counts += [l]
            # print(l + ' mndr')


    #mir2disease
    mir2disease = pd.read_excel('../data/eval_data/mir2disease.xlsx',header=None)[[0,2]]
    mir2disease = mir2disease.dropna(axis=0, subset=[2])
    pick = mir2disease.loc[mir2disease[2].str.contains('pan', case=False)][0].tolist()
    for l in rnalist:
        if l in set(pick):
            rele_counts += [l]
            # print(l + ' mir2disease')

    #oncomirdb
    oncomirdb = pd.read_excel('../data/eval_data/oncomirdb.v-1.1-20131217_download.xls')[['tissue', 'mirbase_r20']]
    pick = oncomirdb.loc[oncomirdb['tissue']=='pancreas']['mirbase_r20'].tolist()
    pick=['hsa'+p for p in pick]
    for l in rnalist:
        if l in set(pick):
            rele_counts += [l]
            # print(l + ' oncomirdb')


    #mircancer
    mircancer = pd.read_excel('../data/eval_data/miRCancerJune2020.txt.xlsx')[['mirId   Cancer', 'Profile PubMed Article']]
    pick = mircancer.loc[mircancer['Profile PubMed Article'].str.contains('pan', case=False)]['mirId   Cancer'].tolist()
    for l in rnalist:
        if l in set(pick):
            rele_counts += [l]
            # print(l + ' mircancer')


    #msdd
    msdd = pd.read_excel('../data/eval_data/msdd.xlsx')[['miRNA', 'Disease']]
    pick = msdd.loc[msdd['Disease'].str.contains('pan', case=False)]['miRNA'].tolist()
    pick = ['hsa' + p for p in pick]
    for l in rnalist:
        if l in set(pick):
            rele_counts += [l]
            # print(l + ' msdd')
    #mndr
    # 旧
    # mndr=pd.read_excel('../data/eval_data/MNDR_Experimental lncRNA-disease information.xlsx')[['ncRNA Symbol','Disease']]
    # 新
    mndr=pd.read_excel('../data/eval_data_new/RNADiseasev4.0_RNA-disease_experiment_all.xlsx')[['RNA Symbol','Disease Name']]
    # pick=mndr.loc[mndr['Disease'].str.contains('pan', case=False)]['ncRNA Symbol'].tolist()
    pick=mndr.loc[mndr['Disease Name'].str.contains('pan', case=False)]['RNA Symbol'].tolist()
    for l in rnalist:
        for p in set(pick):
            if l in p:
                rele_counts += [l]
                # print(l+' mndr')

    # print(len(set(rele_counts)))
    # return len(set(rele_counts))
    return set(rele_counts)


#25  131  733
#690  168  2359


