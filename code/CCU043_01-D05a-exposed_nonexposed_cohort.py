# Databricks notebook source
# MAGIC %md # CCU043_01-D05a-exposed_nonexposed_cohort
# MAGIC  
# MAGIC **Description** This notebook updates the baseline patient table with covid-19 diagnosis and diabetes records from curated datasets.
# MAGIC  
# MAGIC **Authors** Sharmin Shabnam
# MAGIC
# MAGIC **Reviewers** 
# MAGIC
# MAGIC **Acknowledgements** Tom Bolton, Fionna Chalmers, Anna Stevenson (Health Data Science Team, BHF Data Science Centre). Based on CCU002_07 and subsequently CCU003_05-D04-skinny
# MAGIC
# MAGIC **Notes**
# MAGIC
# MAGIC **Data Output**
# MAGIC - **`ccu043_01_tmp_baseline_cohort_with_covid_diabetes`** : Temp baseline cohort with covid-19 and diabetes records
# MAGIC
# MAGIC **Notes**

# COMMAND ----------

spark.sql('CLEAR CACHE')
spark.conf.set('spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation', 'true')

# COMMAND ----------

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

# MAGIC %run "/Shared/SHDS/common/functions"

# COMMAND ----------

# MAGIC %md # 0 Parameters

# COMMAND ----------

# MAGIC %run "./CCU043_01-D01-parameters"

# COMMAND ----------

# MAGIC %md # 1 Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Baseline Cohort

# COMMAND ----------

tmp_cohort = spark.table(path_tmp_baseline)
count_var(tmp_cohort, 'person_id')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 COVID

# COMMAND ----------

covid = spark.table(path_cur_covid)                    
mapping= {
        'chess': '1.admission',
        'hes_apc': '1.admission',
        'gdppr': '2.diagnosis',
        'sgss': '2.diagnosis',
        'death': '3.death'
    }
covid = (covid
          .withColumn("source_ori", f.col("source"))
          .replace(to_replace=mapping, subset=['source'])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Diabetes

# COMMAND ----------

diab = spark.table(path_cur_diabetes)

# COMMAND ----------

diab_bhf = (spark.table(path_cur_diabetes_bhf)
            .select('PERSON_ID','out_diabetes','date_of_diagnosis')
            .withColumnRenamed('PERSON_ID', 'person_id')
            .withColumnRenamed('out_diabetes', 'diab_name_bhf')
            .withColumnRenamed('date_of_diagnosis', 'diab_date_bhf')
            .withColumn('diab_name_bhf', 
                              f.when(f.col('diab_name_bhf') == 'Type 1', 'diabetes_type1')
                               .when(f.col('diab_name_bhf') == 'Type 2', 'diabetes_type2')
                               .when(f.col('diab_name_bhf') == 'Other', 'diabetes_other')
                               .otherwise(f.col('diab_name_bhf')))
            .filter(f.col('diab_date_bhf').isNotNull())
            )

# COMMAND ----------

# MAGIC %md
# MAGIC # 2 Add Diabetes Dates

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Diab first Code algorithm

# COMMAND ----------

diab = (diab
        .join(tmp_cohort
              .select('person_id', 'sex', 'censor_start_date', 'censor_end_date'),
              'person_id','left')
        )
count_var(diab, 'person_id')
null_counts = {col:diab.filter(diab[col].isNull()).count() for col in diab.columns}
null_counts

# COMMAND ----------

diab = (diab
        .where(f.col('censor_start_date').isNotNull()) 
        .where((f.col('diab_date') >= f.col('censor_start_date')) 
               & (f.col('diab_date') <= f.col('censor_end_date')))
        .drop('diab_diag_position')
        )

count_var(diab, 'person_id')
tmpt = tab(diab,'diab_source')
tmpt = tab(diab,'diab_name')
null_counts = {col:diab.filter(diab[col].isNull()).count() for col in diab.columns}
null_counts

# COMMAND ----------

diab = (diab
        .withColumn("diab_name",
                    f.when(
                      ((f.col("sex") == "M") & (f.col("diab_name") == "diabetes_gestational")),
                       f.lit(None))
                    .otherwise(f.col("diab_name"))
                             )
        .filter(f.col('diab_name').isNotNull())
        .drop('sex')
)
count_var(diab, 'person_id')
tmpt = tab(diab,'diab_source')
tmpt = tab(diab,'diab_name')

# COMMAND ----------

_win = Window.partitionBy(['person_id']).orderBy('diab_date')  
diab = (diab
        .orderBy('person_id', 'diab_source')
        .withColumn('_rownum', f.row_number().over(_win))
        .where(f.col('_rownum') == 1)
        .drop('censor_start_date', 'censor_end_date', '_rownum', 'diab_code_type')
        .orderBy('person_id', 'diab_date')
        )
tmpt = tab(diab,'diab_source')
tmpt = tab(diab,'diab_name')
count_var(diab, 'person_id')

# COMMAND ----------

tmp_cohort = (tmp_cohort
        .join(diab, 'person_id','left')
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 BHF Diab Algorithm

# COMMAND ----------

diab_bhf = (diab_bhf
        .join(tmp_cohort
              .select('person_id','censor_start_date', 'censor_end_date'),
              'person_id','left')
        )
count_var(diab_bhf, 'person_id')
null_counts = {col:diab_bhf.filter(diab_bhf[col].isNull()).count() for col in diab_bhf.columns}
null_counts

# COMMAND ----------

diab_bhf = (diab_bhf
        .where(f.col('censor_start_date').isNotNull()) 
        .where((f.col('diab_date_bhf') >= f.col('censor_start_date')) 
               & (f.col('diab_date_bhf') <= f.col('censor_end_date')))
        )
count_var(diab_bhf, 'person_id')
tmpt = tab(diab_bhf,'diab_name_bhf')
null_counts = {col:diab_bhf.filter(diab_bhf[col].isNull()).count() for col in diab_bhf.columns}
null_counts

# COMMAND ----------

tmp_cohort = (tmp_cohort
        .join(diab_bhf.drop('censor_start_date', 'censor_end_date'), 'person_id','left')
        )
count_var(tmp_cohort, 'person_id')
tmpt = tab(tmp_cohort,'diab_name')
tmpt = tab(tmp_cohort,'diab_name_bhf')

# COMMAND ----------

# MAGIC %md
# MAGIC # 3 Add COVID Dates

# COMMAND ----------

covid_first = (covid
        .orderBy('person_id', 'date')
        .drop('clinical_code', 'description', 'covid_status', 'code')
        .dropDuplicates()
        .dropDuplicates(subset=['person_id', 'date', 'source'])
        .orderBy('person_id', 'date')
        )
count_var(covid_first, 'person_id'); print()

# COMMAND ----------

covid_first = (covid_first
        .join(tmp_cohort
              .select('person_id','censor_start_date', 'censor_end_date'),
              'person_id','left')
        )
count_var(covid_first, 'person_id')
null_counts = {col:covid_first.filter(covid_first[col].isNull()).count() for col in covid_first.columns}
null_counts

# COMMAND ----------

covid_first = (covid_first
        .where(f.col('censor_start_date').isNotNull()) 
        .where((f.col('date') >= f.col('censor_start_date')) 
                 & (f.col('date') <= f.col('censor_end_date')))
        .drop('censor_start_date', 'censor_end_date')
        )
        
count_var(covid_first, 'person_id')
null_counts = {col:covid_first.filter(covid_first[col].isNull()).count() for col in covid_first.columns}
null_counts

# COMMAND ----------

# define windows to calculate rownumbers and date differences with patient/outcome record sets
# _win1 creates a window to order records within same date by source consistently
# _win2 creates a window to order a set of patient records consistently
_win1 = Window\
  .partitionBy('person_id', 'date')\
  .orderBy('source')
_win2 = Window\
  .partitionBy('person_id')\
  .orderBy('date', '_row1')

# add row number and date differences
covid_first = (covid_first
               .withColumn('_row1', f.row_number().over(_win1))
               .withColumn('_row2', f.row_number().over(_win2))
               .withColumn('_diff', f.datediff(f.col('date'), f.lag(f.col('date'), 1).over(_win2)))
               .withColumn('_diff_cumsum', f.sum(f.col('_diff')).over(_win2))
               .drop('_diff')
               .drop('covid_phenotype')
)
#display(covid_first)

# set washout period
washout_period = 28

# restrict to covid infection records within set washout period
covid_first = (covid_first
               .where((f.col('_row2') == 1) | (f.col('_diff_cumsum') <= washout_period))
               .withColumn('flag_hosp',
                            f.when(f.col('source').isin(
                                ['1.admission']),f.lit(1))
                            .otherwise(f.lit(0)))
               .withColumn('flag_hosp_date',
                            f.when(f.col('source').isin(
                                ['1.admission']),f.lit(f.col('date')))
                            .otherwise(f.lit(None)))
               .withColumn('flag_hosp_max', f.max(f.col('flag_hosp')).over(Window.partitionBy('person_id')))
               .withColumn('flag_hosp_date_min', f.min(f.col('flag_hosp_date')).over(Window.partitionBy('person_id')))
               .sort(['person_id', 'date'])
               .drop('_row1')
)
# select first covid infection record only
# select and name final required variables
covid_first = (covid_first
               .where(f.col('_row2') == 1)
               .select('person_id', 'date', 'flag_hosp_max', 'flag_hosp_date_min', 'source', 'source_ori')
               .withColumnRenamed('date', 'covid_first_date')
               .withColumnRenamed('flag_hosp_max', 'covid_first_hosp')
               .withColumnRenamed('flag_hosp_date_min', 'covid_first_hosp_date')
               .withColumnRenamed('source', 'covid_first_type')
               .withColumnRenamed('source_ori', 'covid_first_source')
               .withColumn('covid_first', f.lit(1))
                              )

# COMMAND ----------

count_var(covid_first, 'person_id'); print()
tmpt = tab(covid_first, 'covid_first', var2_unstyled=1); print()
tmpt = tab(covid_first, 'covid_first_hosp', var2_unstyled=1); print()
tmpt = tab(covid_first, 'covid_first_source', var2_unstyled=1); print()
tmpt = tabstat(covid_first, 'covid_first_hosp_date', date=1); print()

# COMMAND ----------

tmp_cohort = (tmp_cohort
        .join(covid_first, 'person_id','left')
        )
count_var(tmp_cohort, 'person_id'); print()
tmpt = tab(tmp_cohort, 'covid_first', var2_unstyled=1); print()
tmpt = tab(tmp_cohort, 'covid_first_hosp', var2_unstyled=1); print()
tmpt = tab(tmp_cohort, 'covid_first_source', var2_unstyled=1); print()
tmpt = tabstat(tmp_cohort, 'covid_first_hosp_date', date=1); print()

# COMMAND ----------

null_counts = {col:tmp_cohort.filter(tmp_cohort[col].isNull()).count() for col in tmp_cohort.columns}
null_counts

# COMMAND ----------

# MAGIC %md
# MAGIC # 5 Save

# COMMAND ----------

save_table(df=tmp_cohort, out_name=f'{proj}_tmp_baseline_cohort_with_covid_diabetes', save_previous=False)
tmp_cohort = spark.table(f'{dsa}.{proj}_tmp_baseline_cohort_with_covid_diabetes')