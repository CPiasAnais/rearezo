import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
#from Categ_Special import plot_Categ_Special, plot_Categ_Y, plotpie, plotGroupResistance, plotCombinedResistance, plotMO, plot_percentage_bool
import seaborn as sns
import pathlib 
import time
import jupyterthemes as jt

st.set_page_config(layout="wide")
jt.jtplot.style(context='paper',theme='grade3')

palette=sns.color_palette("Set2")+sns.color_palette("Set3")+sns.color_palette("Set1")
st.image('./rearezo_logo.png')    
col1, col2=st.columns((4,1))  


def num_to_mean_and_median(df,listcol,timescale):
    newlistcol=[]
    for col in listcol:
        newlistcol.append(col[col.find("_")+1:])
    newlistcol=listcol   
    
   
    gbmean=df.loc[:,newlistcol].groupby(df[timescale]).agg(lambda x: x.mean())
    gbmedian=df.loc[:,newlistcol].groupby(df[timescale]).agg(lambda x: x.median())
    gbmean=gbmean.add_prefix("mean_")
    gbmedian=gbmedian.add_prefix("median_")
    gb=pd.concat([gbmean,gbmedian],axis=1)
    return gb

def add_month_quarter(df,date_col):
    df["MOIS"]=df.loc[:,date_col].map(lambda x: x.strftime('%m-%Y') if pd.notna(x) else np.NaN)
    df["TRIMESTRE"]=df.loc[:,date_col].dt.to_period("Q")
    return df

@st.cache_data
def read_df(filename,date=None):
    if date==None: 
        globals()[filename]=pd.read_csv("./data/"+filename+".csv",low_memory=False)
    else:
        globals()[filename]=pd.read_csv("./data/"+filename+".csv",parse_dates=[date],dayfirst=True,low_memory=False)#,sep=sep,parse_dates=date_columns,dayfirst=True)
        globals()[filename]=add_month_quarter(globals()[filename],date)
    return globals()[filename]

PAT=read_df("./OneHot_Prepared/PAT","DISCHARGE").copy(deep=True)

INF=read_df("./OneHot_Prepared/INF","DATEIN").copy(deep=True)

SERV=read_df("./OneHot_Prepared/SERV").copy(deep=True)

CATH=read_df("./OneHot_Prepared/CATH","CCDATE").copy(deep=True)

MO=read_df("./OneHot_Prepared/MO","DATEIN").copy(deep=True)




#PAT=add_month_quarter(PAT,"SORTIE")
#INF=add_month_quarter(INF,"DATEIN")
#CATH=add_month_quarter(CATH, "DEBUTCC")
#MO=add_month_quarter(MO,"DATEIN")


PAT_BOOL=['SEX', 'DECEASED', 'INTUB', 'PNE', 'CC', 'URI', 'BAC', 'CRI',
                              'REINT', 'IUC', 'ATBADM', 'TRAUMA', 'COLCC', 
                              "IDEP_No Immunodepression", 'IDEP_Other Immunodepression',
                              'IDEP_Applasia <500PN', 'ADMTYPE_Medical', 
                              'ADMTYPE_Scheduled Surgery', 'ADMTYPE_Emergency Surgery',
                              'ORIGIN01_Community', 'ORIGIN01_Short term care units', 
                              'ORIGIN01_Long Term / Ward in units', 
                              'ORIGIN01_ICU', 'ORIGIN95_Hospital (other than ICU)', 
                              'ORIGIN95_ICU', 'ORIGIN95_Community', 'ECMO_Not infected', 
                              'ECMO_Veino-Veinous', 'ECMO_Veino-Arterial', 'COVID_Not infected',
                              'COVID_Confirmed', 'COVID_Possible']

PAT_NUM_NAME=["AGE","INTUBLEN","IUCLEN","DURATION","IGSII","DIFFERENCE_ADM_INTUBDATE",
                      "DIFFERENCE_ADM_IUCDATE"]

PAT_NUM=['mean_AGE', 'median_AGE', 'mean_INTUBLEN', 'median_INTUBLEN', 'mean_IUCLEN',
         'median_IUCLEN', 'mean_DURATION', 'median_DURATION', 'mean_IGSII', 'median_IGSII',
         'mean_DIFFERENCE_ADM_INTUBDATE', 'median_DIFFERENCE_ADM_INTUBDATE', 
         'mean_DIFFERENCE_ADM_IUCDATE', 'median_DIFFERENCE_ADM_IUCDATE']

INF_BOOL=['AMC1', 'AMC2', 'AMP1', 'AMP2', 'CAZ1', 'CAZ2', 'BLSE1', 'BLSE2', 'C3G1', 'C3G2',
          'CAR1', 'CAR2', 'PANR1', 'PANR2', 'COL1', 'COL2', 'GLY1', 'GLY2', 'OXA1', 'OXA2',
          'PTZ1', 'PTZ2', 'FLU1', 'FLU2', 'PANR - Confirmed1', 'PANR - Confirmed2','EPC1','EPC2',
          'BSISOURCE17_Pulmonary Tract', 'BSISOURCE17_CVC', 'BSISOURCE17_ECMO', 'BSISOURCE17_Digestive Tract',
          'BSISOURCE17_Other Vascular Dispositive','BSISOURCE17_Arteriel Catheter','BSISOURCE17_HDC',
          'BSISOURCE17_Skin / Soft Tissues', 'BSISOURCE17_Urinary Tract', 'BSISOURCE17_PICC',
          'BSISOURCE17_others', 'BSISOURCE17_CCI', 'BSISOURCE17_Peripheral Catheter', 'BSISOURCE17_Osteoarticular',
          'BSISOURCE03_Pulmonary Tract', 'BSISOURCE03_Digestive Tract', 'BSISOURCE03_Catheter', 'BSISOURCE03_Other',
           'BSISOURCE03_Urinary Tract', 'BSISOURCE03_Skin / Soft Tissue', 'CRITYPE_Local CRI',
            'CRITYPE_General CRI', 'CRITYPE_Unspecifed CRI (2,3 or 4)', 'CRITYPE_CLABSI',
            'DCP_Protected Distant Sample','DCP_Unprotected D.S.','DCP_Non Quantitative Aspiration',
             'DCP_Alternative', 'DCP_No Microbiological Criteria', 'SITE_PNE',
              'SITE_URI', 'SITE_BAC', 'SITE_CVC']

CATH_NUM_NAME=['CCLEN', 'DIFFERENCE_CCDATE_CRIDATE']

CATH_NUM=["mean_"+i for i in CATH_NUM_NAME] + ["median_"+i for i in CATH_NUM_NAME]

CATH_BOOL=['AMC1', 'AMC2', 'AMP1', 'AMP2', 'CAZ1', 'CAZ2', 'BLSE1', 'BLSE2', 'C3G1', 'C3G2', 'CAR1', 
           'CAR2', 'PANR1', 'PANR2', 'COL1', 'COL2', 'GLY1', 'GLY2', 'OXA1', 'OXA2', 'PTZ1', 'PTZ2', 
           'FLU1', 'FLU2', 'PANR - Confirmed1', 'PANR - Confirmed2','EPC1','EPC2', 'CCTYPE_CVC', 'CCTYPE_HDC', 
           'CCTYPE_PICC', 'CCSITE_Subclavian', 'CCSITE_Internal Jugular', 'CCSITE_Femoral', 
           'CCSITE_Peripheral', 'CCSITE_Other', 'CRITYPE_Colonisation', 'CRITYPE_Not specified Infection',
           'CRITYPE_Local Infection', 'CRITYPE_General Infection', 'CRITYPE_CRBSI', 
           "CRITYPE_No Infection nor colonisation", 'LABO_Sent to the laboratory for analysis', 'LABO_Not removed dureing the stay',
           'LABO_Not Cultured']

SERV_NUM_NAME= ['LITETAB', 'LITSERV', 'NBADM', 'NBADMJ', 'NB_ADMIS_SURV', 'JHSHA', 'CONSOSHA']

SERV_NUM =["mean_"+i for i in SERV_NUM_NAME] + ["median_"+i for i in SERV_NUM_NAME]
SERV_BOOL=['ENVOIATB', 'ENVOIABR', 'DECONTORAL', 'DDSGASTR', 'DDSIV', 'DDSOROPH', 'STATSERV_ICU',
            'STATSERV_Step-down unit', 'STATSERV_Other Intensive Care', 'TYPETAB_CH', 'TYPETAB_CHU',
            'TYPETAB_MIL', 'TYPETAB_MCO', 'TYPETAB_AUT', 'TYPETAB_CAC', 'STATETAB_PUB', 'STATETAB_PRI',
            'STATETAB_PIC', 'STATETAB_INC', 'STATETAB_ESPI', 'TYPSERV_Mixed', 'TYPSERV_Surgery',
            'TYPSERV_Medical', 'TYPSERV_Neurology', 'TYPSERV_Cardiology', 'TYPSERV_Burnt', 
            'INFORMAT_No', 'INFORMAT_Fully', 'INFORMAT_Partially', 'DDS_No','DDS_Some patients']


@st.cache_data
def to_timecode(scale):
        """  --- input: a word for the time scale to use
             --- output: its 'offset alias' code, as defined in pandas 'pd.Series.dt.to_period' """
        if scale=='Year':
            return 'Y'
        if scale== 'Quarter':
            return 'Q'
        if scale =='Month':
            return 'M'
        if scale=='Week':
            return 'W'
        
        
def remove_prefix(list1):
    return [i[4:] for i in list1]

def MO_name_to_MO_code(molist,mo_codes):
    """Permet la conversion d'un nom de MO à son nom"""
    ENT=["ENT1NS", "CITFRE","CITKOS","CITAUT","ENTAER","ENTCLO","ENTNSP","ENTAUT","ESCCOL","HAFSPP","ENT2NS","KLEOXY","KLEPNE","ENT3NS",
    "KLEAUT","MOGSPP","PRTMIR","PRTAUT","PRVSPP","SALTYP","SALAUT","SERSPP","SHISPP","ETBAUT","ENTAUT"]
    ENT3G=[]#"KLEAUT","MOGSPP","PRTMIR","PRTAUT","PRVSPP","SALTYP","SALAUT","SERSPP","SHISPP"]
    KLE=["KLEOXY","KLEPNE","KLEAUT","KLENSP"]
    PRT=["PRTMIR","PRTAUT","PRTNSP"]
    ENTEROBACTER=["ENTAER","ENTCLO"]
    
    for mo in molist: 
        
        if mo=="Enterobacter Cloacae & Aerogenes":
            for c in ENTEROBACTER:
                mo_codes.append(c)
        
        if mo=='Pseudomonas Aeruginosa':
            mo_codes.append('PSEAER')
        if mo=='Acinetobacter Bau.':
            mo_codes.append('ACIBAU')
        if mo=='Ent. Faecalis':
            mo_codes.append('ENCFAE')
            
        if mo =='Ent. Faecium':
            mo_codes.append('ENCFAC')
            
        if mo =='Staph. Aureus':
            mo_codes.append('STAAUR')
        if mo=='Enterobacteriaceae':
            for c in ENT:
                mo_codes.append(c)
        if mo=='IIIrd Group Enterobacteriaceae':
            for c in ENT3G:
                mo_codes.append(c)
        if mo=='Klebsellia':
            for c in KLE:
                mo_codes.append(c)
        if mo=='Proteus':
            for c in PRT:
                mo_codes.append(c)
        if mo=='Serratia':
            mo_codes.append('SERSPP')

    return mo_codes

@st.cache_data
def groupall_PAT(PAT,timescale): 

    #"""Groupe toutes les variables du dataset patient pour l'échelle temporelle donnée """                               
    grouped_pat_bool_ALL=PAT.groupby(PAT[timescale])[PAT_BOOL].agg(lambda x: 100*x.sum()/x.count() if x.count()>0 else np.NaN)#.add_prefix("PAT_")
    grouped_pat_num_ALL=num_to_mean_and_median(PAT,PAT_NUM_NAME, timescale)
    return grouped_pat_bool_ALL, grouped_pat_num_ALL
@st.cache_data
def groupall_CATH(CATH,timescale):       
                     
    grouped_cath_bool_ALL=CATH.groupby(CATH[timescale])[CATH_BOOL].agg(lambda x: 100*x.sum()/x.count() if x.count()>0 else np.NaN)#.add_prefix("CATH_")
    grouped_cath_num_ALL=num_to_mean_and_median(CATH,CATH_NUM_NAME, timescale)
    return grouped_cath_bool_ALL, grouped_cath_num_ALL
@st.cache_data
def groupall_SERV(SERV,timescale):                        
    grouped_serv_bool_ALL=SERV.groupby(SERV[timescale])[SERV_BOOL].agg(lambda x: 100*x.sum()/x.count() if x.count()>0 else np.NaN)#.add_prefix("SERV_")
    grouped_serv_num_ALL=num_to_mean_and_median(SERV,SERV_NUM_NAME, timescale)
    return grouped_serv_bool_ALL, grouped_serv_num_ALL

@st.cache_data
def groupall_INF(INF,timescale):                              
    grouped_inf_bool_ALL=INF.groupby(INF[timescale])[INF_BOOL].agg(lambda x: 100*x.sum()/x.count() if x.count()>0 else np.NaN)#.add_prefix("INF_")
   
    return grouped_inf_bool_ALL

        
rcol=["AMC1","AMC2","AMP1","AMP2","CAZ1","CAZ2","BLSE1","BLSE2","C3G1","C3G2","CAR1","CAR2","PANR1",
      "PANR2","COL1","COL2","GLY1","GLY2","OXA1","OXA2","PTZ1","PTZ2","FLU1","FLU2","PANR - Confirmed1",
      "PANR - Confirmed2","EPC1","EPC2"]

rcol_MO=[i.replace("1","") for i in rcol]
rcol_cat=["CAT_"+i for i in rcol] 
rcol_inf=["INF_"+i for i in rcol]  

with col1:
    
    st.title("Rea-Rezo : Visualisation de tendances")
    
    st.write("Cette petite app locale est faite pour visualiser facilement les données issues des BD Rea-Rezo 1995 - 2022")
    
    #INFECTION=pd.read_csv("../BD_convert/Final/INFECTION.csv",sep=";",parse_dates=["DATEIN"])
    #INFECTION.TYPEILC.replace({2:'Local Infection',3:'General infection',4:'CLABSI',5:np.NaN},inplace=True)
    #PATIENT=pd.read_csv("../BD_convert/Final/PATIENT.csv",sep=",",parse_dates=["ENTREE","SORTIE","DEBUTINTUB","FININTUB","DEBUTVC","FINVC","DEBUTSAD","FINSAD"])
    empty=pd.DataFrame()
    @st.cache()
    def keepdf(a):  
        return globals()[a]
    
    
    with st.sidebar:
        #a=st.selectbox('Which DataFrame would you like to work on ?',  ["Infection","Patient","Service","Catheter"])
        
        st.title("Paramètres des graphes")
       # timescale=st.select_slider("Selectionner une granularité temporelle",options=['ANNEE','TRIMESTRE','MOIS'])
        
        timescale='YEAR'
        filter_by_service=st.expander(label="Sélection des services")
        select_serv=filter_by_service.checkbox("Sélectionner uniquement les données de certain services ?")    
            
        grouped_datasets=pd.DataFrame()
        #grouped_datasets=#read_df("./grouped/grouped_"+timescale.lower()).copy(deep=True)
        grouped_filtered=pd.DataFrame()
        grouped_filtered_pat=pd.DataFrame()
        grouped_filtered_inf=pd.DataFrame()
        grouped_filtered_serv=pd.DataFrame()
        grouped_filtered_cath=pd.DataFrame()
        #if not filter_by_service:
        #    grouped_datasets=read_df("./grouped/grouped_"+timescale.lower()).copy(deep=True)
        #    print(grouped_datasets)
        #    grouped_datasets=grouped_datasets.groupby("ANNEE").agg(lambda x: 100*x.sum()/x.SER_mean_NB_ADMIS_SURV)
        #    grouped_datasets.reset_index(inplace=True)
        #    grouped_datasets.set_index("ANNEE",inplace=True)

        if select_serv:
            selected_services=st.multiselect("Les données de quel(s) service(s) souhaitez-vous visualiser ?" ,PAT.SERVICE.unique())
            #grouped_datasets=read_df("./grouped/grouped_"+timescale.lower()+"_service").copy(deep=True)
            #grouped_datasets=grouped_datasets.loc[grouped_datasets.SERVICE.isin(selected_services)]
            PAT=PAT.loc[PAT.SERVICE.isin(selected_services)]
            CATH=CATH.loc[CATH.SERVICE.isin(selected_services)]
            INF=INF.loc[INF.SERVICE.isin(selected_services)]
            SERV=SERV.loc[SERV.SERVICE.isin(selected_services)]
            MO=MO.loc[MO.SERVICE.isin(selected_services)]
            #grouped_datasets.set_index("ANNEE")stre
            #MO= ## RAJOTUER IDEM POUR MO 
            
        
            #df.set_index("ANNEE",inplace=True)
            

        

        ## Variables générales : 
            
        grouped_num=pd.DataFrame()
        grouped_bool=pd.DataFrame()

        grouped_inf_bool_ALL=groupall_INF(INF,timescale)
        grouped_serv_bool_ALL, grouped_serv_num_ALL=groupall_SERV(SERV,"YEAR")
        
            
        grouped_pat_bool_ALL, grouped_pat_num_ALL=groupall_PAT(PAT,timescale)
        
        grouped_cath_bool_ALL, grouped_cath_num_ALL=groupall_CATH(CATH,timescale)
        #grouped_pat_ALL=pd.concat([grouped_pat_bool_ALL,grouped_pat_num_ALL],axis=1)
        
        
            
        #"""analysis= st.radio("Quel type de courbes voulez vous tracer ?",["Evolution Temporelle","Ecologie bactérienne"])
        
        #if analysis=="Evolution Temporelle":     """
        
        st.title("Courbes à visualiser :")  
        ## ---------------------------PATIENT ----------
        filtered_pat_col_bool=[]
        filtered_pat_col_num=[]
        grouped_filtered_pat_num_ALL=pd.DataFrame()        
        grouped_filtered_pat_bool_ALL=pd.DataFrame() 
        
        pat_exp=st.expander(label="Dataset Patient")
        
        
        pat_col=pat_exp.multiselect("Quelles colonnes du dataset Patient souhaitez-vous visualiser ?",options=PAT_BOOL+PAT_NUM)
        
        pat_col_bool=[i for i in pat_col if i in PAT_BOOL]
        pat_col_num=[i for i in pat_col if i in PAT_NUM]
                   
             
        
        
        pat_col_plot=pat_col_bool+pat_col_num
        
        filter_pat_bool=pat_exp.checkbox("Filtrer certaines variables (pat) ?")
        filtered_pat_col= []
        
        
        if filter_pat_bool:                
            #PAT_filtered=PAT.copy(deep=True)
            filter_pat_bool=True
            pat_exp.markdown("**Filtrage des colonnes Patient**")
            filtered_pat_col=pat_exp.multiselect("Quelles colonnes souhaitez vous filtrer (pat) ?", pat_col)
            
            if len(filtered_pat_col)>0:                    
                filter_pat_col=pat_exp.selectbox("Par quelle colonne souhaitez-vous filtrer (pat) ?",[i for i in PAT.columns if i.find("_")<0])
                filter_pat_value=pat_exp.multiselect("Quelles valeurs souhaitez vous garder (pat) ?", PAT[filter_pat_col].unique())
                
                filtered_pat_data=PAT.loc[PAT[filter_pat_col].isin(filter_pat_value)]
                
                grouped_filtered_pat_bool_ALL,grouped_filtered_pat_num_ALL=groupall_PAT(filtered_pat_data,timescale)

                grouped_filtered_pat_bool_ALL=grouped_filtered_pat_bool_ALL.add_suffix(" (filtré)")
                grouped_filtered_pat_num_ALL=grouped_filtered_pat_num_ALL.add_suffix(" (filtré)")
                
                filtered_pat_col_bool=[i for i in filtered_pat_col if i in PAT_BOOL]
                filtered_pat_col_num=list(i for i in filtered_pat_col if i in PAT_NUM)
                
                
                
                if pat_exp.checkbox("Cacher la courbe de la variable non filtrée ?",key='pat_keep'):
                    pat_col_bool=[i for i in pat_col_bool if i not in filtered_pat_col_bool]
                    pat_col_num=[i for i in pat_col_num if i not in filtered_pat_col_num]
                
                
                filtered_pat_col_bool=[str(i) + " (filtré)" for i in filtered_pat_col_bool]
                filtered_pat_col_num=[str(i) + " (filtré)" for i in filtered_pat_col_num]
                
  
        
        ## ---------------------------INFECTION ----------
        filtered_inf_col_bool=[]
        
        grouped_filtered_inf_num_ALL=pd.DataFrame()        
        grouped_filtered_inf_bool_ALL=pd.DataFrame() 
        
        inf_exp=st.expander(label="Dataset Infection")
        
        
        inf_col=inf_exp.multiselect("Quelles colonnes du dataset Infection souhaitez-vous visualiser ?",options=[i  for i in INF_BOOL if i not in rcol])
        
        inf_col_bool=[i for i in inf_col if i in INF_BOOL]  
                   
        inf_col_plot=inf_col_bool
        
        filter_inf_bool=inf_exp.checkbox("Filtrer certaines variables (inf) ?")
        filtered_inf_col= []
        
        
        if filter_inf_bool:                
            #INF_filtered=INF.copy(deep=True)
            filter_inf_bool=True
            inf_exp.markdown("**Filtrage des colonnes Infection**")
            filtered_inf_col=inf_exp.multiselect("Quelles colonnes souhaitez vous filtrer (inf) ?", inf_col)
            
            if len(filtered_inf_col)>0:                    
                filter_inf_col=inf_exp.selectbox("Par quelle colonne souhaitez-vous filtrer (inf) ?",[ i for i in INF.columns if (i.find("_")<0)&(i not in rcol)])
                filter_inf_value=inf_exp.multiselect("Quelles valeurs souhaitez vous garder (inf) ?", INF[filter_inf_col].unique())
                
                filtered_inf_data=INF.loc[INF[filter_inf_col].isin(filter_inf_value)]
                
                grouped_filtered_inf_bool_ALL=groupall_INF(filtered_inf_data,timescale).add_suffix(" (filtré)")

                
                
                filtered_inf_col_bool=[i for i in filtered_inf_col if i in INF_BOOL]
                
                if inf_exp.checkbox("Cacher la courbe de la variable non filtrée ?",key='inf_keep'):
                    inf_col_bool=[i for i in inf_col_bool if i not in filtered_inf_col_bool]
                
                               
                
                filtered_inf_col_bool=[str(i) + " (filtré)" for i in filtered_inf_col_bool]
                
               
                    
                    
        ## ---------------------------CATHETER ----------
        filtered_cath_col_bool=[]
        filtered_cath_col_num=[]
        grouped_filtered_cath_num_ALL=pd.DataFrame()        
        grouped_filtered_cath_bool_ALL=pd.DataFrame() 
        
        cath_exp=st.expander(label="Dataset Cathéter")
        
        
        cath_col=cath_exp.multiselect("Quelles colonnes du dataset Cathéter souhaitez-vous visualiser ?",options=[i for i in CATH_BOOL+CATH_NUM if i not in rcol])
        
        cath_col_bool=[i for i in cath_col if i in CATH_BOOL]
        cath_col_num=[i for i in cath_col if i in CATH_NUM]
                   
             
        
        
        cath_col_plot=cath_col_bool+cath_col_num
        
        filter_cath_bool=cath_exp.checkbox("Filtrer certaines variables (cath) ?")
        filtered_cath_col= []
        
        
        if filter_cath_bool:                
            #CATH_filtered=CATH.copy(deep=True)
            filter_cath_bool=True
            cath_exp.markdown("**Filtrage des colonnes Cathéter**")
            filtered_cath_col=cath_exp.multiselect("Quelles colonnes souhaitez vous filtrer (cath) ?", cath_col)
            
            if len(filtered_cath_col)>0:                    
                filter_cath_col=cath_exp.selectbox("Par quelle colonne souhaitez-vous filtrer (cath) ?",[i for i in CATH.columns if (i.find("_")<0)&(i not in rcol)])
                filter_cath_value=cath_exp.multiselect("Quelles valeurs souhaitez vous garder (cath) ?", CATH[filter_cath_col].unique())
                
                filtered_cath_data=CATH.loc[CATH[filter_cath_col].isin(filter_cath_value)]
                
                grouped_filtered_cath_bool_ALL,grouped_filtered_cath_num_ALL=groupall_CATH(filtered_cath_data,timescale)

                grouped_filtered_cath_bool_ALL=grouped_filtered_cath_bool_ALL.add_suffix(" (filtré)")
                grouped_filtered_cath_num_ALL=grouped_filtered_cath_num_ALL.add_suffix(" (filtré)")
                
                filtered_cath_col_bool=[i for i in filtered_cath_col if i in CATH_BOOL]
                
                filtered_cath_col_num=list(i for i in filtered_cath_col if i in CATH_NUM)
                
                if cath_exp.checkbox("Cacher la courbe de la variable non filtrée ?",key='cath_keep'):
                    cath_col_num=[i for i in cath_col_num if i not in filtered_cath_col_num]
                    cath_col_bool=[i for i in cath_col_bool if i not in filtered_cath_col_bool]
                
                filtered_cath_col_bool=[str(i) + " (filtré)" for i in filtered_cath_col_bool]
                filtered_cath_col_num=[str(i) + " (filtré)" for i in filtered_cath_col_num]
  
                
                     
        ## ---------------------------SERVICE --------------------
        filtered_serv_col_bool=[]
        filtered_serv_col_num=[]
        grouped_filtered_serv_num_ALL=pd.DataFrame()        
        grouped_filtered_serv_bool_ALL=pd.DataFrame() 
        
        serv_exp=st.expander(label="Dataset Service")
            
        
        serv_col=serv_exp.multiselect("Quelles colonnes du dataset Service souhaitez-vous visualiser ?",options=SERV_BOOL+SERV_NUM)
        
        serv_col_bool=[i for i in serv_col if i in SERV_BOOL]
        serv_col_num=[i for i in serv_col if i in SERV_NUM]
                   
             
        
        
        serv_col_plot=serv_col_bool+serv_col_num
        
        filter_serv_bool=serv_exp.checkbox("Filtrer certaines variables (serv) ?")
        filtered_serv_col= []
        
        
        if filter_serv_bool:                
            #SERV_filtered=SERV.copy(deep=True)
            filter_serv_bool=True
            serv_exp.markdown("**Filtrage des colonnes Service**")
            filtered_serv_col=serv_exp.multiselect("Quelles colonnes souhaitez vous filtrer (serv) ?", serv_col)
            
            if len(filtered_serv_col)>0:                    
                filter_serv_col=serv_exp.selectbox("Par quelle colonne souhaitez-vous filtrer (serv) ?",[i for i in SERV.columns if i.find("_")<0])
                filter_serv_value=serv_exp.multiselect("Quelles valeurs souhaitez vous garder (serv) ?", SERV[filter_serv_col].unique())
                
                filtered_serv_data=SERV.loc[SERV[filter_serv_col].isin(filter_serv_value)]
                
                grouped_filtered_serv_bool_ALL,grouped_filtered_serv_num_ALL=groupall_SERV(filtered_serv_data,timescale)

                grouped_filtered_serv_bool_ALL=grouped_filtered_serv_bool_ALL.add_suffix(" (filtré)")
                grouped_filtered_serv_num_ALL=grouped_filtered_serv_num_ALL.add_suffix(" (filtré)")
                
                filtered_serv_col_bool=[i for i in filtered_serv_col if i in SERV_BOOL]
                
                
                filtered_serv_col_num=list(i for i in filtered_serv_col if i in SERV_NUM)
                
                
                if serv_exp.checkbox("Cacher la courbe de la variable non filtrée ?"):
                    serv_col_num=[i for i in serv_col_num if i not in filtered_serv_col_num]
                    serv_col_bool=[i for i in serv_col_bool if i not in filtered_serv_col_bool]
                
                filtered_serv_col_bool=[str(i) + " (filtré)" for i in filtered_serv_col_bool]
                filtered_serv_col_num=[str(i) + " (filtré)" for i in filtered_serv_col_num]
                
               

                       
                
        ### --------------------RESISTANCES -----------------------------------------------------
 
        grouped_res=pd.DataFrame()
        res_exp=st.expander(label="Résistances")
        
        list_of_MOlist={}
        list_of_MO_codes={}
        antibiotics={}
        
        nb_MO=int(res_exp.number_input("Les résistances de combien de groupes de MO voulez-vous tracer ?",min_value=0,step=1))
        for i in range(1,nb_MO+1):
            
            res_exp.markdown("** Groupe "+str(i)+"**")
            MOlist=res_exp.multiselect("(Pour le "+str(i)+"eme groupe :) De quels micro-organismes souhaitez-vous visualiser les résistances ?",help="Si vous en séléctionnez plusieurs, il s'agira de la résistance du groupe ainsi créé",key="mo_list_"+str(i),options=["Ent. Faecalis","Ent. Faecium","Enterobacteriaceae","Enterobacter Cloacae & Aerogenes","Pseudomonas Aeruginosa",'Staph. Aureus',"Acinetobacter Bau.","Klebsellia","Proteus","Serratia",'IIIrd Group Enterobacteriaceae'])
            MO_codes=[]
            
            MO_name_to_MO_code(MOlist, MO_codes)
            
            nb_autre_MO=int(res_exp.number_input("Ajouter X micro-organismes (si absents de la liste ci-dessus)",key="nb_autre_MO"+str(i),help="Entrez le code à 6 lettres du micro-organisme",min_value=0,step=1))
            if nb_autre_MO >0:
                for j in range(nb_autre_MO):
                    add_any=res_exp.text_input('Ajouter le code à 6 lettres du '+str(j+1)+'e MO supplémentaire',key="autreMO"+str(j))
                    MO_codes.append(add_any)
            list_of_MOlist[i]=MOlist
            list_of_MO_codes[i]=MO_codes
            antibiotics[i]=res_exp.multiselect("(Pour le  "+str(i)+" eme groupe ainsi formé) Les résistances à quel(s) antibiotique(s) souhaitez-vous visualiser ?",key="mo_list_"+str(i),options=['OXA','C3G','GLY',  'AMC', 'AMP', 'CAR', 'CAZ', 'COL', 'PTZ', 'PANR', 'FLU', 'BLSE','PANR - Confirme'])
            
            grouped_res=pd.concat([grouped_res,MO.loc[MO.MOIN.isin(MO_codes)].groupby(timescale)[antibiotics[i]].agg(lambda x: 100*x.mean()).add_prefix("Groupe_"+str(i)+"_R_à_")],axis=1)
            
        
        grouped_filtered_res=pd.DataFrame()
        filter_res_bool=res_exp.checkbox("Filtrer certaines résistances ?")
        
        if filter_res_bool  :
            res_exp.markdown("**Filtrage des résistances**")
            filtered_res_col=res_exp.multiselect("Les résistances de quel(s) groupe(s) de MO souhaitez-vous filtrer ?", list_of_MOlist.keys(),format_func=lambda x:"Groupe "+str(x))
            if len(filtered_res_col)>0:
                filter_res_col=res_exp.selectbox("Par quel attribut souhaitez-vous filtrer ces résistances ?",[i for i in MO.columns if (i.find("_")<0)&(i not in rcol_MO)])
                filter_res_value=res_exp.multiselect("Quelles valeurs souhaitez vous garder ?", MO[filter_res_col].unique())
                
                
                filtered_MO_data=MO.loc[MO[filter_res_col].isin(filter_res_value)]
                
                for i in filtered_res_col:
                   
                    
                    
                    grouped_filtered_res=pd.concat([grouped_filtered_res,filtered_MO_data.loc[filtered_MO_data.MOIN.isin(list_of_MO_codes[i])].groupby(timescale)[antibiotics[i]].agg(lambda x: 100*x.mean()).add_prefix("Groupe_"+str(i)+"_à_").add_suffix(" (filtré)")],axis=1)
                    
      
        
                        
                        

    grouped_num=pd.concat([grouped_num,grouped_pat_num_ALL.loc[:,pat_col_num],grouped_filtered_pat_num_ALL.loc[:,filtered_pat_col_num],grouped_cath_num_ALL.loc[:,cath_col_num],grouped_filtered_cath_num_ALL.loc[:,filtered_cath_col_num],grouped_serv_num_ALL.loc[:,serv_col_num],grouped_filtered_serv_num_ALL.loc[:,filtered_serv_col_num]],axis=1)
    grouped_bool=pd.concat([grouped_bool,grouped_res,grouped_filtered_res,grouped_pat_bool_ALL.loc[:,pat_col_bool],grouped_filtered_pat_bool_ALL.loc[:,filtered_pat_col_bool],grouped_inf_bool_ALL.loc[:,inf_col_bool],grouped_filtered_inf_bool_ALL.loc[:,filtered_inf_col_bool],grouped_serv_bool_ALL.loc[:,serv_col_bool],grouped_filtered_serv_bool_ALL.loc[:,filtered_serv_col_bool],grouped_cath_bool_ALL.loc[:,cath_col_bool],grouped_filtered_cath_bool_ALL.loc[:,filtered_cath_col_bool]],axis=1)



            
    
    
        
    with col1:
        
        all_plot_columns= pat_col_plot + inf_col_plot +serv_col_plot+cath_col_plot
        
        all_filtered_columns = filtered_pat_col+filtered_inf_col+filtered_cath_col + filtered_serv_col 
        not_filtered_columns=[i for i in all_plot_columns if i not in all_filtered_columns]
        
        
        all_bool_col=pat_col_bool+inf_col_bool+cath_col_bool+serv_col_bool+filtered_pat_col_bool+filtered_inf_col_bool+filtered_cath_col_bool+filtered_serv_col_bool
        all_num_col=pat_col_num+cath_col_num+serv_col_num+filtered_pat_col_num+filtered_cath_col_num+filtered_serv_col_num
        
        all_filtered_bool_col=filtered_pat_col_bool+filtered_inf_col_bool+filtered_cath_col_bool+filtered_serv_col_bool
        all_filtered_num_col=filtered_pat_col_num+filtered_cath_col_num+filtered_serv_col_num
        
        all_notfiltered_bool_col=pat_col_bool+inf_col_bool+cath_col_bool+serv_col_bool
        all_notfiltered_num_col=pat_col_num+cath_col_num+serv_col_num
        
        
                    
        legend=True        
    
        
         
        
        # Présentation du graphe : 
        title=st.text_input("Saisir le titre du graphe",value='Titre du graphe')
        
        graph_params=st.expander(label='Paramètres du graphe')
        
        fig,ax = plt.subplots(figsize=(10,6),tight_layout=True)   
        #plt.figure()   
        ax.set_xlabel('Year')
        
        ax.set_ylabel('Percent')
        ax.set_title(title,fontsize=17)
        #ax.set_xticks(np.arange(1995,2020,2))
        graph_params.markdown("** Début et fin ** ")
        custom_debut_fin=graph_params.checkbox("Changer les dates de début et de fin ?")
        if custom_debut_fin:
            debut= graph_params.number_input('Début', min_value=1995, max_value=2029, step=1)
            fin= graph_params.number_input('Fin', min_value=1996, max_value=2030, step=1,value=2020)
            ax.set_xlim([debut,fin])
        #if not custom_debut_fin:ax.set_xlim([1995,2020])
        graph_params.markdown("** Minima / Maxima des axes**")
        custom_min_max_y=graph_params.checkbox("Changer le minimum et maximum des ordonnées pour les variables en pourcentage ?")
        if custom_min_max_y: 
            min_y=graph_params.number_input('Valeur minimum des ordonnées', min_value=0., max_value=99.99,step=0.01)
            max_y=graph_params.number_input('Valeur maximum des ordonnées', min_value=0.01, max_value=100.,step=0.01,value=100.)
            ax.set_ylim([min_y,max_y])
        
        options={"linestyle":'--',"marker":'o',"color":palette}
        if len(all_bool_col)>0:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())   
            bool_plot=grouped_bool.plot(ax=ax,**options)
            #ax.legend(title=titlelabel,loc=0,fancybox=True)#,bbox_to_anchor=(1.05,1.05))
            
            if len(all_num_col)>0:  
                num=ax.twinx()
                num.set_ylabel("Nombre réel (triangles)")
                options_num={"linestyle":'--',"marker":'^',"color":palette[len(all_bool_col):]}
                num_plot=grouped_num.plot(ax=num,**options_num)
                #num.legend(title="Variabes numériques ",loc=4    ,fancybox=True)
                #ax.legend(title="Variables en pourcentages",loc=1   ,fancybox=True)
                
                #graph_min_max=graph_params.(label="Changer les valeurs minimales et maximales des axes ?")
                custom_min_max_z=graph_params.checkbox("Changer le minimum et maximum des ordonnées pour les variables numériques ?")
                if custom_min_max_z: 
                    min_z=graph_params.number_input('Valeur minimum des ordonnées (numérique)', min_value=0., max_value=99.99,step=0.01)
                    max_z=graph_params.number_input('Valeur maximum des ordonnées ( numérique)', min_value=0.1      , max_value=1000.,step=0.1,value=100.)
                    num.set_ylim([min_z,max_z])
                num.get_legend().remove()
            ax.get_legend().remove()    
        
        #st.write(PAT.MOIS.apply(lambda x: x.to_timestamp if pd.notna(x) else np.NaN))
        
        

        graph_params.markdown("**Grille**")
        grid_bool=graph_params.checkbox("Afficher la grille pour les pourcentages ?",value=True)
        if not grid_bool: ax.grid(False)
        if len(all_num_col)>0:
            grid_num=graph_params.checkbox("Afficher la grille pour les valeurs numériques ?")
            if not grid_num:num.grid(False)
        
        #labels= all_notfiltered_bool_col + all_filtered_bool_col + all_notfiltered_num_col +  all_filtered_num_col 
        
        
        graph_params.markdown("** Légende ** ")
        titlelabel=graph_params.text_input("Saisir le titre de la légende",value='Légende') 
        
        outside=graph_params.checkbox("Placer la légende en dehors du graphe ? ")
        if outside: 
            fig.set_size_inches(11,6)
            fig.legend(title=titlelabel,fancybox=True,bbox_to_anchor=(1.1,1))
        if not outside: fig.legend(title=titlelabel,fancybox=True)#,bbox_to_anchor=(1.2,1))
        
        options={"linestyle":'--',"marker":'o',"color":palette}
        #options_filter={"linestyle":'--',"marker":'o',"color":palette[len(plot_columns)+len(antibiotics):]}
        
        #labels= antibiotics + [ i[4:] + "_filtré" for i in all_filtered]  + [ i[4:] for i in plot_columns]
    
        #if len(all_filtered)>0:
            #grouped_filtered.plot(ax=ax,**options_filter)
            
        #if plot_res:
            
        #    res_grouped.plot(ax=ax,**options_atb)
        #st.write( MO.loc[MO.MOIN.isin(MO_codes),antibiotics].groupby("ANNEE").mean())
        #if len(plot_columns)>0:
        #    df.loc[:,plot_columns].plot(ax=ax,**options)
        
        
            #grouped_pat_bool.plot(ax=ax,**options)
            #grouped_pat_num.plot(ax=num,**options)
        #plt.grid(which='major',color='lightgray',linestyle='--', linewidth=1)   
        
        
        #num.set_ylabel('Valeur Numérique')
        #st.write( df.loc[:,inf_col+pat_col+ser_col+cat_col])#.plot(ax=plt.gca(),linestyle='--',marker='o',color=sns.color_palette('Set2')))
        #plt.tight_layout()  
        #   st.write(df)#fig=plt.gca()
        #if st.button("Mettre à jour le graphe "):
        if timescale=="YEAR":
            st.pyplot(fig)
        display_table=st.expander(label='Données du graphe')
        display_table.write(grouped_bool.join(grouped_num))
        

        
        if len(list_of_MOlist)>0:
                display_MO_detail=st.expander(label="Détail des groupes de MO")
                resistances_df=pd.DataFrame(index=["Groupe " +str(i) + ": " for i in list(list_of_MOlist.keys())])
                
                resistances_df["MOs du groupe"]=list_of_MO_codes.values()
                resistances_df["antibiotiques"]=antibiotics.values()
                display_MO_detail.write(resistances_df)
        #plot_data=pd.concat([df.loc[:,plot_columns],grouped_filtered,res_grouped],axis=1)
        #st.write(plot_data)


with col2:
    st.title("Sauvegarde")
    #st.markdown("Cela peut prendre un petit peu de temps selon la taille des données, pensez à bien attendre que l'opération soit terminée avant de poursuivre")
    if st.button('Sauvegarder le graphe ',help='Le graphe sera dans le dossier ./SAVED/courbes en csv, avec le titre donné au graphe'):    
        fig.savefig('./SAVED/courbes/'+title+'.png',bbox_inches='tight')
        

    
    
    if st.button('Sauvegarder les données',help='Le fichier sera dans le dossier ./SAVED/data en csv, avec le titre donné au graphe'):   
        #st.markdown(get_table_download_link(plot_data), unsafe_allow_html=True)
        grouped_bool.join(grouped_num).to_csv('./SAVED/data/'+title+'.csv',index=True,sep=";")
        
        
           