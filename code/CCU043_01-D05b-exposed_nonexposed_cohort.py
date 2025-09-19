# Databricks notebook source
# MAGIC %md # CCU043_01-D05b-exposed_nonexposed_cohort
# MAGIC  
# MAGIC **Description** This notebook creates the exposed and non-exposed cohort from the baseline table.
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
# MAGIC - **`ccu043_01_exposed_cohort`** : Exposed cohort (those with covid-19 and prevalent diabetes)
# MAGIC - **`ccu043_01_nonexposed_cohort`** : Non-exposed cohort (all individuals in the baseline cohort)

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

tmp_cohort = spark.table(f'{dsa}.{proj}_tmp_baseline_cohort_with_covid_diabetes')
count_var(tmp_cohort, 'person_id')

# COMMAND ----------

# MAGIC %md
# MAGIC # 4 Create Exposed Non-Exposed Cohorts

# COMMAND ----------

idr = tmp_cohort.filter(f.col('region').isNull()).select('person_id').filter(f.col('person_id').isNotNull()).distinct().count()
idi = tmp_cohort.filter(f.col('imd_quintile').isNull()).select('person_id').filter(f.col('person_id').isNotNull()).distinct().count()
print(idr, idi)

# COMMAND ----------

tmp1 = (tmp_cohort
        .where(f.col('region').isNotNull())
        .where(f.col('imd_quintile').isNotNull())
        .orderBy('date_of_birth')
        .withColumn('yob', f.year(f.col('date_of_birth')))
        .select(['person_id','yob', 'sex', 
                 'region', 'imd_quintile',
                 'date_of_death',
                 'diab_date',
                 'diab_name',
                 'diab_name_bhf',
                 'diab_date_bhf',
                 'covid_first', 'covid_first_date', 'covid_first_hosp', 'covid_first_hosp_date',
                 'censor_start_date', 'censor_end_date'])
        .fillna(0, subset=['covid_first'])
        )
tmpt = tab(tmp1, 'diab_name_bhf', var2_unstyled=1); print()
tmp1 = (tmp1
        .withColumn('diab_date_bhf', 
                              f.when(f.col('diab_name_bhf') == 'Unlikely', f.lit(None))
                               .otherwise(f.col('diab_date_bhf'))
                              )
        .withColumn('diab_name_bhf', 
                              f.when(f.col('diab_name_bhf') == 'Unlikely', f.lit(None))
                               .otherwise(f.col('diab_name_bhf'))
                              )
        .withColumn('diab_name_bhf', 
                              f.when(f.col('diab_name_bhf') == 'NOS', 'diabetes_other')
                               .otherwise(f.col('diab_name_bhf'))
                              )
)
tmpt = tab(tmp1, 'diab_name_bhf', var2_unstyled=1); print()
count_var(tmp1, 'person_id')

# COMMAND ----------

tmpt = tab(tmp1, 'yob', var2_unstyled=1); print()
tmpt = tab(tmp1, 'region', var2_unstyled=1); print()
tmpt = tab(tmp1, 'imd_quintile', var2_unstyled=1); print()
tmpt = tab(tmp1, 'sex', var2_unstyled=1); print()
tmpt = tab(tmp1, 'covid_first', var2_unstyled=1); print()
tmpt = tab(tmp1, 'covid_first_hosp', var2_unstyled=1); print()
tmpt = tab(tmp1, 'diab_name', var2_unstyled=1); print()
tmpt = tab(tmp1, 'diab_name_bhf', var2_unstyled=1); print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 COVID - 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1.1 Diab First Code Cohort

# COMMAND ----------

exp_covid = (tmp1
        .where(f.col('covid_first') == 1)
        .drop('covid_first', 'covid_first_hosp', 'hosp', 'hosp_date','censor_start_date', 'censor_end_date', 'covid_first_hosp_date')
        )
count_var(exp_covid, 'person_id'); print()
id1 = exp_covid.select('person_id').filter(f.col('person_id').isNotNull()).distinct().count()
tmpt = tab(exp_covid,'diab_name')

exp_covid = (exp_covid
        .where((f.col('diab_date').isNull()) | (f.col('diab_date') > f.col('covid_first_date')))
        .withColumnRenamed('covid_first_date', 'index_date')
        )
count_var(exp_covid, 'person_id'); print()
id21 = exp_covid.select('person_id').filter(f.col('person_id').isNotNull()).distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 NONEXPOSED - 1

# COMMAND ----------

nonexp = (tmp1
        .drop(
              'covid_first',
              'covid_first_hosp', 'covid_first_hosp_date', 
              'hosp', 'hosp_date',
              'censor_start_date', 'censor_end_date'
              )
        )
count_var(nonexp, 'person_id'); print()
id5 = nonexp.select('person_id').filter(f.col('person_id').isNotNull()).distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3 Save

# COMMAND ----------

save_table(df=exp_covid, out_name=f'{proj}_exposed_cohort', save_previous=False)
exp_covid = spark.table(f'{dsa}.{proj}_exposed_cohort')

save_table(df=nonexp, out_name=f'{proj}_nonexposed_cohort', save_previous=False)
nonexp = spark.table(f'{dsa}.{proj}_nonexposed_cohort')

# COMMAND ----------

schema = t.StructType([t.StructField('step', t.StringType(), True),
                       t.StructField('count', t.StringType(), True),
                      t.StructField('context', t.StringType(), True),]
                       )
flow_chart = spark.createDataFrame([], schema)
step0 = [('total patients with missing region: ', idr, '')]
step00 = [('total patients with missing imd: ', idi, '')]
step1 = [('total patients in the exposed cohort - with covid: ', id1, '')]
step2 = [('total patients in the exposed cohort - with covid and wo prev diab: ', id21, '')]
step3 = [('total patients in the nonexposed cohort', id5, '')]

flow_chart = (flow_chart
              .union(spark.createDataFrame(step0, schema))
              .union(spark.createDataFrame(step00, schema))
              .union(spark.createDataFrame(step1, schema))
              .union(spark.createDataFrame(step2, schema))
              .union(spark.createDataFrame(step3, schema))
             )
schema = flow_chart.schema
pandas_df = flow_chart.toPandas()
pd.set_option('display.max_rows', len(pandas_df))
pd.set_option('display.max_columns', len(pandas_df.columns))
pd.set_option('expand_frame_repr', False) 
print(pandas_df)

def custom_round(x, base=5):
    return int(base * round(float(x)/base))
pandas_df['count'] = pandas_df['count'].apply(lambda x: custom_round(x, base=5)).astype(int).apply('{:,}'.format)
 
print(pandas_df)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('expand_frame_repr')      

flow_chart = spark.createDataFrame(pandas_df,schema=schema)
display(flow_chart)  