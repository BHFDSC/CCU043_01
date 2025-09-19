# Databricks notebook source
# MAGIC %md # CCU043_01-D04-baseline_cohort
# MAGIC  
# MAGIC **Description** This notebook creates the baseline patient table from skinny data produce by the BHF DSC, which includes key patient characteristics.
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
# MAGIC - **`ccu043_01_tmp_baseline_cohort`** : Temp baseline cohort before applying prevalent diabetes criteria
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

skinny = spark.table(path_tmp_skinny)

# COMMAND ----------

# MAGIC %md # 2 Check Skinny Data

# COMMAND ----------

count_var(skinny, 'person_id')
id1 = skinny.select('person_id').filter(f.col('person_id').isNotNull()).distinct().count()

# COMMAND ----------

tmpt = tab(skinny, 'ethnicity_5_group', var2_unstyled=1); print()
tmpt = tab(skinny, 'ethnicity_19_group', var2_unstyled=1); print()
tmpt = tab(skinny, 'sex', var2_unstyled=1); print()
tmpt = tab(skinny, 'region', var2_unstyled=1); print()
tmpt = tab(skinny, 'imd_quintile', var2_unstyled=1); print()
tmpt = tab(skinny, 'in_gdppr', var2_unstyled=1); print()

# COMMAND ----------

null_counts = {col:skinny.filter(skinny[col].isNull()).count() for col in skinny.columns}
null_counts

# COMMAND ----------

# MAGIC %md
# MAGIC # 3 Create Baseline Cohort

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 GDPPR

# COMMAND ----------

baseline_cohort = (skinny
              .filter(f.col("in_gdppr")==1)
              .drop('sex_code', 'ethnicity_raw_code', 'ethnicity_raw_description', 'ethnicity_19_code', 'ethnicity_19_group', 'in_gdppr','imd_decile', 'date_of_birth_tie_flag', 'sex_tie_flag', 'ethnicity_19_code_tie_flag', 'lsoa_tie_flag')
)

count_var(baseline_cohort, 'person_id')
id2 = baseline_cohort.select('person_id').filter(f.col('person_id').isNotNull()).distinct().count()

# COMMAND ----------

tmpt = tab(baseline_cohort, 'ethnicity_5_group', var2_unstyled=1); print()
tmpt = tab(baseline_cohort, 'sex', var2_unstyled=1); print()
tmpt = tab(baseline_cohort, 'region', var2_unstyled=1); print()
tmpt = tab(baseline_cohort, 'imd_quintile', var2_unstyled=1); print()

# COMMAND ----------

null_counts = {col:baseline_cohort.filter(baseline_cohort[col].isNull()).count() for col in baseline_cohort.columns}
null_counts

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Alive

# COMMAND ----------

start_date = '2020-01-01'
end_date   = '2024-05-28'
print(start_date, end_date)

# COMMAND ----------

tmp = (baseline_cohort
              .filter((f.col("date_of_death") < f.col("date_of_birth")))
              )
count_var(tmp, 'person_id')
display(tmp)

# COMMAND ----------

# ==========================================
#             Baseline Cohort 2020-2024 
# ==========================================
baseline_cohort = (baseline_cohort
                   .withColumn('study_start_date', f.to_date(f.lit(start_date)))
                   .withColumn('study_end_date', f.to_date(f.lit(end_date)))
                   .withColumn('age_at_start', f.round(f.datediff(f.col('study_start_date'), f.col('date_of_birth'))/365.25, 2))
                   .filter((f.col("date_of_death").isNull()) | (f.col("date_of_death") > f.col('study_start_date')))
                   .filter((f.col("age_at_start") >= 18) & (f.col("age_at_start") <= 110))
                   .orderBy('date_of_birth')
                   .withColumn('censor_start_date', f.date_add(f.col('date_of_birth'), -0))
                   .withColumn('censor_end_date',
                      f.when(
                        (f.col('date_of_death').isNotNull()), f.col('date_of_death'))
                      .otherwise(f.col('study_end_date'))
                     )
              )

count_var(baseline_cohort, 'person_id')
id3 = baseline_cohort.select('person_id').filter(f.col('person_id').isNotNull()).distinct().count()

# COMMAND ----------

tmpt = tab(baseline_cohort, 'ethnicity_5_group', var2_unstyled=1); print()
tmpt = tab(baseline_cohort, 'sex', var2_unstyled=1); print()
tmpt = tab(baseline_cohort, 'region', var2_unstyled=1); print()
tmpt = tab(baseline_cohort, 'imd_quintile', var2_unstyled=1); print()

# COMMAND ----------

null_counts = {col:baseline_cohort.filter(baseline_cohort[col].isNull()).count() for col in baseline_cohort.columns}
null_counts

# COMMAND ----------

id4 = baseline_cohort.filter((f.col("sex").isNull()) | (f.col("sex")=='I')).select('person_id').filter(f.col('person_id').isNotNull()).distinct().count()
id5 = baseline_cohort.filter(f.col('date_of_birth').isNull()).select('person_id').filter(f.col('person_id').isNotNull()).distinct().count()
print(id4, id5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Age, Sex, LSOA

# COMMAND ----------

# ==========================================
#             Baseline Cohort 2020-2024 
# ==========================================
baseline_cohort = (baseline_cohort
              .filter(f.col("date_of_birth").isNotNull())
              #.filter(f.col("lsoa").isNotNull())
              .filter((f.col("sex").isNotNull()) & (f.col("sex")!='I'))
)  

count_var(baseline_cohort, 'person_id')
id6 = baseline_cohort.select('person_id').filter(f.col('person_id').isNotNull()).distinct().count()

# COMMAND ----------

tmpt = tab(baseline_cohort, 'ethnicity_5_group', var2_unstyled=1); print()
tmpt = tab(baseline_cohort, 'sex', var2_unstyled=1); print()
tmpt = tab(baseline_cohort, 'region', var2_unstyled=1); print()
tmpt = tab(baseline_cohort, 'imd_quintile', var2_unstyled=1); print()
tmpt = tab(baseline_cohort, 'death_flag', var2_unstyled=1); print()

# COMMAND ----------

null_counts = {col:baseline_cohort.filter(baseline_cohort[col].isNull()).count() for col in baseline_cohort.columns}
null_counts

# COMMAND ----------

count_var(baseline_cohort, 'person_id')
display(baseline_cohort)

# COMMAND ----------

# MAGIC %md ## 3.4 Save

# COMMAND ----------

save_table(df=baseline_cohort, out_name=f'{proj}_tmp_baseline_cohort', save_previous=False)
spark.sql(f"""REFRESH TABLE {path_tmp_baseline}""")

baseline_cohort = spark.table(f'{dsa}.{proj}_tmp_baseline_cohort')
count_var(baseline_cohort, 'person_id')
display(baseline_cohort)

# COMMAND ----------

tmpt = tabstat(baseline_cohort, 'age_at_start', date=0); print()
tmpt = tabstat(baseline_cohort, 'age_at_start', byvar='sex', date=0); print()
tmpt = tabstat(baseline_cohort, 'date_of_birth', date=1); print()

# COMMAND ----------

schema = t.StructType([t.StructField('step', t.StringType(), True),
                       t.StructField('count', t.StringType(), True),
                      t.StructField('context', t.StringType(), True),]
                       )
flow_chart = spark.createDataFrame([], schema)
step1 = [('total patients overall: ', id1, '')]
step2 = [('total registered in GDPPR: ', id2, '')]
step3 = [('total alive and registered on 01/01/2020: N = ', id3, '')]
step4 = [('total with missing sex: ', id4, '')]
step5 = [('total with missing age: ', id5, '')]
step6 = [('total without missing age, sex: ', id6, '')]

flow_chart = (flow_chart
              .union(spark.createDataFrame(step1, schema))
              .union(spark.createDataFrame(step2, schema))
              .union(spark.createDataFrame(step3, schema))
              .union(spark.createDataFrame(step4, schema))
              .union(spark.createDataFrame(step5, schema))
              .union(spark.createDataFrame(step6, schema))
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