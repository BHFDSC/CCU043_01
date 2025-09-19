# Databricks notebook source
# MAGIC %md 
# MAGIC # CCU043_01_D03b_curated_data_covid
# MAGIC
# MAGIC
# MAGIC **Description** This notebook creates the covid phenotypes table of CCU043_01.
# MAGIC
# MAGIC **Author(s)** Tom Bolton, Fionna Chalmers, Anna Stevenson - Health Data Science Team, BHF Data Science Centre
# MAGIC
# MAGIC **Reviewers** 
# MAGIC
# MAGIC **Acknowledgements** Based on CCU007_05, CCU002_07 and subsequently CCU051_D03e_curated_data_covid
# MAGIC
# MAGIC **Data Output**
# MAGIC - **`ccu043_01_cur_covid`** : Covid phenotypes for each person

# COMMAND ----------

spark.sql('CLEAR CACHE')
spark.conf.set('spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation', 'true')

# COMMAND ----------

# DBTITLE 1,Libraries
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window

from functools import reduce

import databricks.koalas as ks
import pandas as pd
import numpy as np

import re
import io
import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns

print("Matplotlib version: ", matplotlib.__version__)
print("Seaborn version: ", sns.__version__)
_datetimenow = datetime.datetime.now() # .strftime("%Y%m%d")
print(f"_datetimenow:  {_datetimenow}")

# COMMAND ----------

# DBTITLE 1,Functions
# MAGIC %run "/Shared/SHDS/common/functions"

# COMMAND ----------

# MAGIC %md # 0. Parameters

# COMMAND ----------

# MAGIC %run "./CCU043_01-D01-parameters"

# COMMAND ----------

start_date = '2020-01-01'
end_date   = '2024-05-28'
print(start_date, end_date)

# COMMAND ----------

# MAGIC %md # 1. Data

# COMMAND ----------

codelist = spark.table(path_out_codelist_covid)

sgss     = extract_batch_from_archive(parameters_df_datasets, 'sgss')
gdppr    = extract_batch_from_archive(parameters_df_datasets, 'gdppr')
hes_apc  = extract_batch_from_archive(parameters_df_datasets, 'hes_apc')
chess    = extract_batch_from_archive(parameters_df_datasets, 'chess')
deaths_long  = spark.table(path_cur_deaths_long)

# COMMAND ----------

display(codelist)

# COMMAND ----------

# MAGIC %md # 2. Prepare

# COMMAND ----------

_hes_apc = hes_apc\
  .select(['PERSON_ID_DEID', 'EPISTART', 'DIAG_4_01', 'DIAG_4_CONCAT', 'OPERTN_4_CONCAT', 'SUSRECID'])\
  .withColumnRenamed('PERSON_ID_DEID', 'PERSON_ID')\
  .withColumnRenamed('EPISTART', 'DATE')\
  .where(f.col('DIAG_4_CONCAT').rlike('U07(3|4)'))\
  .where((f.col('DATE') >= start_date) & (f.col('DATE') <= end_date))\
  .dropDuplicates()

count_var(_hes_apc, 'PERSON_ID'); print()


# COMMAND ----------

display(_hes_apc)

# COMMAND ----------

# sgss
_sgss = sgss\
  .select(['PERSON_ID_DEID', 'Reporting_Lab_ID', 'Specimen_Date'])\
  .withColumnRenamed('PERSON_ID_DEID', 'PERSON_ID')\
  .withColumnRenamed('Specimen_Date', 'DATE')\
  .where((f.col('DATE') >= start_date) & (f.col('DATE') <= end_date))\
  .dropDuplicates()

# gdppr
# omitted: 'LSOA'
_gdppr = gdppr\
  .select(['NHS_NUMBER_DEID', 'DATE', 'CODE'])\
  .withColumnRenamed('NHS_NUMBER_DEID', 'PERSON_ID')\
  .where((f.col('DATE') >= start_date) & (f.col('DATE') <= end_date))\
  .dropDuplicates()

# hes_apc
# omitted: 'DISMETH', 'DISDEST', 'DISDATE', 'SUSRECID'
_hes_apc = hes_apc\
  .select(['PERSON_ID_DEID', 'EPISTART', 'DIAG_4_01', 'DIAG_4_CONCAT', 'OPERTN_4_CONCAT', 'SUSRECID'])\
  .withColumnRenamed('PERSON_ID_DEID', 'PERSON_ID')\
  .withColumnRenamed('EPISTART', 'DATE')\
  .where(f.col('DIAG_4_CONCAT').rlike('U07(1|2)'))\
  .where((f.col('DATE') >= start_date) & (f.col('DATE') <= end_date))\
  .dropDuplicates()

# chess
_chess = chess\
  .select(['PERSON_ID_DEID', 'Typeofspecimen', 'Covid19', 'AdmittedToICU', 'Highflownasaloxygen', 'NoninvasiveMechanicalventilation', 'Invasivemechanicalventilation', 'RespiratorySupportECMO', 'DateAdmittedICU', 'HospitalAdmissionDate', 'InfectionSwabDate'])\
  .withColumnRenamed('PERSON_ID_DEID', 'PERSON_ID')\
  .withColumnRenamed('InfectionSwabDate', 'DATE')\
  .where(f.col('Covid19') == 'Yes')\
  .where(\
    ((f.col('DATE') >= start_date) | (f.col('DATE').isNull()))\
    & ((f.col('DATE') <= end_date) | (f.col('DATE').isNull()))\
  )\
  .where(\
    ((f.col('HospitalAdmissionDate') >= start_date) | (f.col('HospitalAdmissionDate').isNull()))\
    & ((f.col('HospitalAdmissionDate') <= end_date) | (f.col('HospitalAdmissionDate').isNull()))\
  )\
  .where(\
    ((f.col('DateAdmittedICU') >= start_date) | (f.col('DateAdmittedICU').isNull()))\
    & ((f.col('DateAdmittedICU') <= end_date) | (f.col('DateAdmittedICU').isNull()))\
  )\
  .dropDuplicates()
  
# deaths
_deaths = deaths_long\
  .where((f.col('DATE') >= start_date) & (f.col('DATE') <= end_date))

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Covid positive

# COMMAND ----------

# sgss
# note: all records are included as every record is a "positive test"
# -- TODO: wranglers please clarify whether LAB ID 840 is still the best means of identifying pillar 1 vs 2
# -- CASE WHEN REPORTING_LAB_ID = '840' THEN "pillar_2" ELSE "pillar_1" END as description,
#   .withColumn('description', f.when(f.col('Reporting_Lab_ID') == '840', 'pillar_2').otherwise('pillar_1'))\
_sgss_pos = _sgss\
  .withColumn('covid_phenotype', f.lit('01_Covid_positive_test_sgss'))\
  .withColumn('clinical_code', f.lit(''))\
  .withColumn('description', f.lit(''))\
  .withColumn('covid_status', f.lit(''))\
  .withColumn('code', f.lit(''))\
  .withColumn('source', f.lit('sgss'))\
  .select(f.col('PERSON_ID').alias('person_id'), f.col('DATE').alias('date'), 'covid_phenotype', 'clinical_code', 'description', 'covid_status', 'code', 'source')

# gdppr
# note: need to inspect and identify which are only suspected NOT confirmed!
_codelist_gdppr = codelist\
  .where(f.col('name') == 'covid19')\
  .select(['code', 'term'])

_gdppr_pos = _gdppr\
  .select([f.col('PERSON_ID').alias('person_id'), f.col('DATE').alias('date'), 'CODE'])\
  .join(f.broadcast(_codelist_gdppr), on='code', how='inner')\
  .withColumn('covid_phenotype', f.lit('01_Covid_diagnosis_GDPPR'))\
  .withColumnRenamed('CODE', 'clinical_code')\
  .withColumnRenamed('term', 'description')\
  .withColumn('covid_status', f.lit(''))\
  .withColumn('code', f.lit('SNOMED'))\
  .withColumn('source', f.lit('gdppr'))

# COMMAND ----------

# MAGIC %md # 4. Covid admission

# COMMAND ----------

# ------------------------------------------------------------------------------
# hes_apc
# ------------------------------------------------------------------------------
# any
_hes_apc_adm_any = _hes_apc\
  .where(f.col('DIAG_4_CONCAT').rlike('U07(1|2)'))\
  .withColumn('covid_phenotype', f.lit('02_Covid_admission_anypos'))\
  .withColumn('clinical_code',\
    f.when(f.col('DIAG_4_CONCAT').rlike('U071'), 'U071')\
    .when(f.col('DIAG_4_CONCAT').rlike('U072'), 'U072')\
  )\
  .withColumn('description',\
    f.when(f.col('DIAG_4_CONCAT').rlike('U071'), 'Confirmed_COVID19')\
    .when(f.col('DIAG_4_CONCAT').rlike('U072'), 'Suspected_COVID19')\
  )\
  .withColumn('covid_status',\
    f.when(f.col('DIAG_4_CONCAT').rlike('U071'), 'confirmed')\
    .when(f.col('DIAG_4_CONCAT').rlike('U072'), 'suspected')\
  )\
  .withColumn('code', f.lit('ICD10'))\
  .withColumn('source', f.lit('hes_apc'))\
  .select(f.col('PERSON_ID').alias('person_id'), 'DATE', 'covid_phenotype', 'clinical_code', 'description', 'covid_status', 'code', 'source')\
  .where(f.col('clinical_code')!='U072')

# pri
_hes_apc_adm_pri = _hes_apc\
  .where(f.col('DIAG_4_01').rlike('U07(1|2)'))\
  .withColumn('covid_phenotype', f.lit('02_Covid_admission_pripos'))\
  .withColumn('clinical_code',\
    f.when(f.col('DIAG_4_01').rlike('U071'), 'U071')\
    .when(f.col('DIAG_4_01').rlike('U072'), 'U072')\
  )\
  .withColumn('description',\
    f.when(f.col('DIAG_4_01').rlike('U071'), 'Confirmed_COVID19')\
    .when(f.col('DIAG_4_01').rlike('U072'), 'Suspected_COVID19')\
  )\
  .withColumn('covid_status',\
    f.when(f.col('DIAG_4_01').rlike('U071'), 'confirmed')\
    .when(f.col('DIAG_4_01').rlike('U072'), 'suspected')\
  )\
  .withColumn('code', f.lit('ICD10'))\
  .withColumn('source', f.lit('hes_apc'))\
  .select(f.col('PERSON_ID').alias('person_id'), f.col('DATE').alias('date'), 'covid_phenotype', 'clinical_code', 'description', 'covid_status', 'code', 'source')\
  .where(f.col('clinical_code')!='U072')


# ------------------------------------------------------------------------------
# chess
# ------------------------------------------------------------------------------
_chess_adm = _chess\
  .select([f.col('PERSON_ID').alias('person_id'), f.col('HospitalAdmissionDate').alias('date')])\
  .withColumn('covid_phenotype', f.lit('02_Covid_admission_anypos'))\
  .withColumn('clinical_code', f.lit(''))\
  .withColumn('description', f.lit('HospitalAdmissionDate IS NOT null'))\
  .withColumn('covid_status', f.lit('confirmed'))\
  .withColumn('code', f.lit(''))\
  .withColumn('source', f.lit('chess'))

# COMMAND ----------

# MAGIC %md # 5. Covid death

# COMMAND ----------

deaths_long = deaths_long.select([f.col(x).alias(x.lower()) for x in deaths_long.columns])
display(deaths_long)

# COMMAND ----------

# ==========================================
#              DEATH POSITIVE
# ==========================================
_deaths_pos_any = (deaths_long
                   .where(f.col('code').rlike('U071'))
                   .select('person_id', 'date', 'code')
                   .withColumn('covid_phenotype', f.lit('03_Covid_Death_anypos'))
                   .withColumnRenamed('code', 'clinical_code')
                   .withColumn('description', f.lit(''))
                   .withColumn('covid_status', f.lit('confirmed'))
                   .withColumn('code', f.lit(''))
                   .withColumn('source', f.lit('death'))
                   .select('person_id', 'date', 'covid_phenotype', 'clinical_code', 'description', 'covid_status', 'code', 'source')
            )

_deaths_pos_pri = (deaths_long
                   .where(f.col('diag_position') == 'UNDERLYING')
                   .where(f.col('code').rlike('U071'))
                   .select('person_id', 'date', 'code')
                   .withColumn('covid_phenotype', f.lit('03_Covid_Death_pripos'))
                   .withColumnRenamed('code', 'clinical_code')
                   .withColumn('description', f.lit(''))
                   .withColumn('covid_status', f.lit('confirmed'))
                   .withColumn('code', f.lit(''))
                   .withColumn('source', f.lit('death'))
                   .select('person_id', 'date', 'covid_phenotype', 'clinical_code', 'description', 'covid_status', 'code', 'source')
            )

# COMMAND ----------

# MAGIC %md # 6. Combine

# COMMAND ----------

tmp = _sgss_pos\
  .unionByName(_gdppr_pos)\
  .unionByName(_hes_apc_adm_any)\
  .unionByName(_hes_apc_adm_pri)\
  .unionByName(_deaths_pos_pri)\
  .unionByName(_deaths_pos_any)\
  .unionByName(_chess_adm)

# COMMAND ----------

count_var(tmp, 'person_id'); print()
tmpt = tab(tmp, 'covid_phenotype', 'source', var2_unstyled=1); print()

# COMMAND ----------

tmp = (tmp
        .where(f.col('person_id').isNotNull()) 
        .where(f.col('date').isNotNull()) 
        )
null_counts = {col:tmp.filter(tmp[col].isNull()).count() for col in tmp.columns}
null_counts

# COMMAND ----------

count_var(tmp, 'person_id'); print()
tmpt = tab(tmp, 'covid_phenotype', 'source', var2_unstyled=1); print()

# COMMAND ----------

tmpt = tab(tmp, 'clinical_code', var2_unstyled=1); print()
tmpt = tab(tmp, 'description', var2_unstyled=1); print()
tmpt = tab(tmp, 'covid_status', 'source', var2_unstyled=1); print()
tmpt = tab(tmp, 'code', 'source', var2_unstyled=1); print()

# COMMAND ----------

# MAGIC %md # 7. Save

# COMMAND ----------

save_table(df=tmp, out_name=f'{proj}_cur_covid_all', save_previous=False)
tmp = spark.table(f'{dsa}.{proj}_cur_covid_all')

# COMMAND ----------

# MAGIC %md # 8. Check

# COMMAND ----------

# check combined
count_var(tmp, 'person_id'); print()
tmpt = tab(tmp, 'covid_phenotype', 'source', var2_unstyled=1); print()

# COMMAND ----------

null_counts = {col:tmp.filter(tmp[col].isNull()).count() for col in tmp.columns}
null_counts

# COMMAND ----------

tmp1 = tmp.withColumn('source_pheno', f.concat_ws('_', f.col('source'), f.col('covid_phenotype')))
tmpt = tabstat(tmp1, 'date', byvar='source_pheno', date=1); print()

# COMMAND ----------

# check individual
count_var(_sgss_pos, 'person_id'); print()
count_var(_gdppr_pos, 'person_id'); print()
count_var(_hes_apc_adm_any, 'person_id'); print()
count_var(_hes_apc_adm_pri, 'person_id'); print()
count_var(_chess_adm, 'person_id'); print()
count_var(_deaths_pos_any, 'person_id'); print()
count_var(_deaths_pos_pri, 'person_id'); print()