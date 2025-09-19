# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # CCU043_01-D01-parameters
# MAGIC
# MAGIC **Description** This notebook defines a set of parameters, which is loaded in each notebook in the data curation pipeline, so that helper functions and parameters are consistently available.
# MAGIC
# MAGIC **Authors** Tom Bolton, John Nolan, Elena Raffetti
# MAGIC
# MAGIC **Created on** 20.06.2024
# MAGIC
# MAGIC **Last updated on** 20.05.2025
# MAGIC
# MAGIC **Acknowledgements** Based on previous work by Tom Bolton (John Nolan, Elena Raffetti) for CCU018_01 and the CCU002 sub-projects.

# COMMAND ----------

spark.conf.set('spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation', 'true')

# COMMAND ----------

# MAGIC %run "/Shared/SHDS/common/functions"

# COMMAND ----------

import pyspark.sql.functions as f
import pandas as pd
import re

# COMMAND ----------

# MAGIC %md ## 1. Define parameters

# COMMAND ----------

# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------
proj = 'ccu043_01'

# -----------------------------------------------------------------------------
# Databases
# -----------------------------------------------------------------------------
db = ''
dbc = f''
dsa = f''

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# data frame of datasets
#tmp_archived_on = '2024-03-27'

  
# note: the below is largely listed in order of appearance within the pipeline:  

# reference tables
path_ref_bhf_phenotypes  = 'bhf_cvd_covid_uk_byod.bhf_covid_uk_phenotypes_20210127' 
# path_ref_map_ctv3_snomed = 'dss_corporate.read_codes_map_ctv3_to_snomed'
# path_ref_geog            = 'dss_corporate.ons_chd_geo_listings'
# path_ref_imd             = 'dss_corporate.english_indices_of_dep_v02'
# path_ref_gp_refset       = 'dss_corporate.gpdata_snomed_refset_full' 
path_ref_gdppr_refset    = 'dss_corporate.gdppr_cluster_refset' 
# path_ref_icd10           = 'dss_corporate.icd10_group_chapter_v01'

# curated tables
# path_cur_hes_apc_long      = f'{dsa}.{proj}_cur_hes_apc_all_years_archive_long'
# path_cur_hes_apc_oper_long = f'{dsa}.{proj}_cur_hes_apc_all_years_archive_oper_long'
path_cur_deaths_long       = f'{dsa}.{proj}_cur_deaths_long' 
# path_cur_deaths_sing       = f'{dsa}.{proj}_cur_deaths_sing'
# path_cur_lsoa_region       = f'{dsa}.{proj}_cur_lsoa_region_lookup'
# path_cur_lsoa_imd          = f'{dsa}.{proj}_cur_lsoa_imd_lookup'
path_cur_covid             = f'{dsa}.{proj}_cur_covid_all'
path_cur_diabetes          = f'{dsa}.{proj}_cur_diabetes_all'
path_cur_diabetes_bhf      = f'{dsa}.{proj}_ddsc_cohort_out_2024_05_28'

path_tmp_skinny                           = f'{dsa}.hds_curated_assets_demographics_20240425'
path_tmp_baseline                         = f'{dsa}.{proj}_tmp_baseline_cohort'
# path_tmp_cohort                           = f'{dsa}.{proj}_tmp_cohort'

# out tables
path_out_codelist_covariates        = f'{dsa}.{proj}_out_codelist_covariates'
path_out_codelist_covid             = f'{dsa}.{proj}_out_codelist_covid' #
# path_out_codelist_quality_assurance = f'{dsa}.{proj}_out_codelist_quality_assurance'
# path_out_codelist_expo              = f'{dsa}.{proj}_out_codelist_diabetes'
# path_out_skinny                     = f'{dsa}.{proj}_skinny'
path_out_DM_algo_type               = f'{dsa}.{proj}_out_codelist_dm_diabetes_algorithm'

# COMMAND ----------

# Check latest archived dates for each dataset of interest - checked on 2024-03-28

# tmp = spark.table(f'{dbc}.vaccine_status_{db}_archive')
# vacc = tab(tmp, 'archived_on').iloc[:-1,:]


# tmp = spark.table(f'{dbc}.deaths_{db}_archive')
# death = tab(tmp, 'archived_on').iloc[:-1,:]


# tmp = spark.table(f'{dbc}.gdppr_{db}_archive')
# gdppr = tab(tmp, 'archived_on').iloc[:-1,:]


# tmp = spark.table(f'{dbc}.hes_apc_all_years_archive')
# apc = tab(tmp, 'archived_on').iloc[:-1,:]


# tmp = spark.table(f'{dbc}.hes_op_all_years_archive')
# op = tab(tmp, 'archived_on').iloc[:-1,:]


# tmp = spark.table(f'{dbc}.hes_ae_all_years_archive')
# ae = tab(tmp, 'archived_on').iloc[:-1,:]


# tmp = spark.table(f'{dbc}.hes_cc_all_years_archive')
# cc = tab(tmp, 'archived_on').iloc[:-1,:]


# tmp = spark.table(f'{dbc}.sgss_{db}_archive')
# sgss = tab(tmp, 'archived_on').iloc[:-1,:]


# tmp = spark.table(f'{dbc}.chess_{db}_archive')
# chess = tab(tmp, 'archived_on').iloc[:-1,:]


# tmp = spark.table(f'{dbc}.sus_{db}_archive')
# sus = tab(tmp, 'archived_on').iloc[:-1,:]


# print('vaccine', vacc.archived_on.max())
# print('death', death.archived_on.max())
# print('gdppr', gdppr.archived_on.max())
# print('apc', apc.archived_on.max())
# print('op', op.archived_on.max())
# print('ae', ae.archived_on.max())
# print('cc', cc.archived_on.max())
# print('sgss', sgss.archived_on.max())
# print('chess', chess.archived_on.max())
# print('sus', sus.archived_on.max())

# vaccine 2024-06-04
# death 2024-05-28
# gdppr 2024-05-28
# apc 2024-06-04
# op 2024-06-04
# ae 2024-06-04
# cc 2024-06-04
# sgss 2024-03-27
# chess 2024-05-28
# sus 2022-09-30

# COMMAND ----------

# MAGIC %md ## 2. Function to get data on archived_on date

# COMMAND ----------

arc_date = '2024-06-04'
data = [
    ['deaths',  dbc, f'deaths_{db}_archive',        '2024-05-28', 'DEC_CONF_NHS_NUMBER_CLEAN_DEID','REG_DATE_OF_DEATH']
  , ['gdppr',   dbc, f'gdppr_{db}_archive',         '2024-05-28', 'NHS_NUMBER_DEID',        'DATE']
  , ['hes_apc', dbc, f'hes_apc_all_years_archive',      arc_date, 'PERSON_ID_DEID',         'EPISTART'] 
  , ['hes_op',  dbc, f'hes_op_all_years_archive',       arc_date, 'PERSON_ID_DEID',         'APPTDATE'] 
  , ['hes_ae',  dbc, f'hes_ae_all_years_archive',       arc_date, 'PERSON_ID_DEID',         'ARRIVALDATE'] 
  , ['hes_cc',  dbc, f'hes_cc_all_years_archive',       arc_date, 'PERSON_ID_DEID',         'CCSTARTDATE']    
  , ['vacc',    dbc, f'vaccine_status_{db}_archive',    arc_date, 'PERSON_ID_DEID',         'DATE_AND_TIME']
  , ['sgss',    dbc, f'sgss_{db}_archive',          '2024-03-27', 'PERSON_ID_DEID',         'Specimen_Date']
  , ['chess',   dbc, f'chess_{db}_archive',         '2024-05-28', 'PERSON_ID_DEID',         'InfectionSwabDate']       
  #  , ['sus',     dbc, f'sus_{db}_archive',           '2022-09-30', 'NHS_NUMBER_DEID',        'EPISODE_START_DATE'] 
  #  , ['pmeds',   dbc, f'primary_care_meds_{db}_archive', arc_date, 'Person_ID_DEID',           'ProcessingPeriodDate']
  #  , ['ecds',    dbc, f'lowlat_ecds_all_years_archive',  arc_date, 'PERSON_ID_DEID',                 'ARRIVAL_DATE']    
]
parameters_df_datasets = pd.DataFrame(data,
                                       columns = ['dataset', 'database', 'table', 'archived_on', 'idVar', 'dateVar'])
print('parameters_df_datasets:\n', parameters_df_datasets.to_string())

# COMMAND ----------

# function to extract the batch corresponding to the pre-defined archived_on date from the archive for the specified dataset
from pyspark.sql import DataFrame
def extract_batch_from_archive(_df_datasets: DataFrame, _dataset: str):
  
  # get row from df_archive_tables corresponding to the specified dataset
  _row = _df_datasets[_df_datasets['dataset'] == _dataset]
  
  # check one row only
  assert _row.shape[0] != 0, f"dataset = {_dataset} not found in _df_datasets (datasets = {_df_datasets['dataset'].tolist()})"
  assert _row.shape[0] == 1, f"dataset = {_dataset} has >1 row in _df_datasets"
  
  # create path and extract archived on
  _row = _row.iloc[0]
  _path = _row['database'] + '.' + _row['table']  
  _archived_on = _row['archived_on']  
  print(_path + ' (archived_on = ' + _archived_on + ')')
  
  # check path exists # commented out for runtime
#   _tmp_exists = spark.sql(f"SHOW TABLES FROM {_row['database']}")\
#     .where(f.col('tableName') == _row['table'])\
#     .count()
#   assert _tmp_exists == 1, f"path = {_path} not found"

  # extract batch
  _tmp = spark.table(_path)\
    .where(f.col('archived_on') == _archived_on)  
  
  # check number of records returned
  _tmp_records = _tmp.count()
  print(f'  {_tmp_records:,} records')
  assert _tmp_records > 0, f"number of records == 0"

  # return dataframe
  return _tmp

# COMMAND ----------

# MAGIC %md ## 3. Print defined parameters

# COMMAND ----------

print(f'Project:')
print("  {0:<22}".format('proj') + " = " + f'{proj}') 
print(f'')
print(f'Databases:')
print("  {0:<22}".format('db') + " = " + f'{db}') 
print("  {0:<22}".format('dbc') + " = " + f'{dbc}') 
print("  {0:<22}".format('dsa') + " = " + f'{dsa}') 
print(f'')
print(f'Paths:')
# with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
#  print(df_paths_raw_data[['dataset', 'database', 'table']])
# print(f'')
# print(f'  df_archive')
# print(df_archive[['dataset', 'database', 'table', 'productionDate']].to_string())
print(f'')
tmp = vars().copy()
for var in list(tmp.keys()):
  if(re.match('^path_.*$', var)):
    print("  {0:<22}".format(var) + " = " + tmp[var])    
print(f'')
#print(f'composite_events:')   
#for i, c in enumerate(composite_events):
  #print('  ', i, c, '=', composite_events[c])
#print(f'')
# print(f'Out dates:')
# with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
#  print(df_paths_raw_data[['dataset', 'database', 'table']])
# print(f'')
# print(f'  df_out')
#print(df_out[['dataset', 'out_date']].to_string())
#print(f'')