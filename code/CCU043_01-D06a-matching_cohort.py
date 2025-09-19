# Databricks notebook source
# MAGIC %md # CCU043_01-D06a-matching_cohort
# MAGIC  
# MAGIC **Description** This notebook matches cases to controls with a ratio of 1:n based on a set of key patient characteristics, with an additional requirement that all controls alive as at a set index_date which is specific to each case.
# MAGIC  
# MAGIC **Authors** John Nolan, Sharmin Shabnam - Based on Example control selection process in R by John Nolan, and PySpark by Tom Bolton (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Acknowledgements** 
# MAGIC
# MAGIC **Notes**
# MAGIC
# MAGIC **Data Output**
# MAGIC - **`ccu043_01_final_matched`** : 

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
# MAGIC ## 1.1 Import

# COMMAND ----------

cases = spark.table(f'{dsa}.{proj}_exposed_cohort')
count_var(cases, 'person_id')

print(cases.describe(['yob']).show())
tmpt = tab(cases, 'sex', var2_unstyled=1); print()
tmpt = tab(cases, 'region', var2_unstyled=1); print()
tmpt = tab(cases, 'imd_quintile', var2_unstyled=1); print()
display(cases)

# COMMAND ----------

controls = spark.table(f'{dsa}.{proj}_nonexposed_cohort')
count_var(controls, 'person_id')

print(controls.describe(['yob']).show())
tmpt = tab(controls, 'sex', var2_unstyled=1); print()
tmpt = tab(controls, 'region', var2_unstyled=1); print()
tmpt = tab(controls, 'imd_quintile', var2_unstyled=1); print()
display(controls)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2 Prepare and Match

# COMMAND ----------

# define number of controls required per case
controls_per_case = 3

# define matching variables
match_vars  = ["sex", "yob", "region", 'imd_quintile']

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Cases Active

# COMMAND ----------

# select only required variables from cases dataframe
cases = (cases
         .select(['person_id', 'index_date']+match_vars)
         .orderBy('person_id')
)

# define match id (unique match id for each combination of match vars)
window_spec = Window.orderBy(*match_vars)
cases = (cases
          .orderBy('person_id')
          .withColumn("match_id", f.dense_rank().over(window_spec))
          .orderBy('match_id')
)

# COMMAND ----------

# multiply dataframe by number of controls required and add a control_no indicator
cases_1 = (cases
          .withColumn("control_no", f.explode(f.array([f.lit(i) for i in range(controls_per_case)])))
          .withColumn("control_no", f.col("control_no") + f.lit(1))
          .select(['person_id', 'match_id','control_no', 'index_date'])
          .orderBy('match_id', 'control_no')
)
count_var(cases_1, 'person_id')
display(cases_1)

# COMMAND ----------

# prepare cases and controls by adding a row number (stable random order for controls)
win_cases = Window.partitionBy('match_id').orderBy('control_no',"random_order")
#.orderBy('control_no', 'person_id', 'index_date')
cases_1 = (cases_1
           .withColumn("random_order", f.rand(seed=1234))
           .withColumn('rownum', f.row_number().over(win_cases))
           .drop("random_order")
           .orderBy('match_id', 'rownum')
)

# COMMAND ----------

# create a match_id lookup
distinct_matchid = cases.select(*match_vars, "match_id").distinct().orderBy('match_id')
print(distinct_matchid.count())
# display(distinct_matchid)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Controls Active

# COMMAND ----------

# add matching group id to controls and select potential controls for our cases
# this step will drop any controls that have combinations of the matching criteria that do not appear in the cases
controls_1 = (controls
            .join(distinct_matchid, match_vars, 'left')
            .where(f.col('match_id').isNotNull())
            .withColumnRenamed('person_id', 'control_id' )
            .select('control_id', 'match_id',  'date_of_death', 'diab_date', 'covid_first_date')
            .orderBy('match_id', 'control_id')
)

# add a random number for sorting - best if this was added to underlying controls table saved to database for reproducibility
controls_1 = (controls_1
                   .withColumn('control_rand', f.rand(seed = 100))
                   )
count_var(controls_1, 'control_id')

# COMMAND ----------

win_controls = Window.partitionBy('match_id').orderBy('control_rand')
controls_1 = (controls_1
                   .withColumn('rownum', f.row_number().over(win_controls))
                   .select('control_id', 'match_id', 'rownum', 'control_rand',
                             'date_of_death', 'diab_date', 'covid_first_date')
                   .orderBy('match_id', 'rownum')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##2.3 Save

# COMMAND ----------

save_table(df=cases_1, out_name=f'{proj}_cases_active_e1', save_previous=False)
save_table(df=controls_1, out_name=f'{proj}_controls_active_ne1', save_previous=False) 

# COMMAND ----------

# MAGIC %md
# MAGIC # 3 Join 1:3 (Total matched: ~76%)

# COMMAND ----------

# read in case and control tables
cases_1 = spark.table(f'{dsa}.{proj}_cases_active_e1').orderBy('match_id', 'rownum')
display(cases_1)

# COMMAND ----------

controls_1 = spark.table(f'{dsa}.{proj}_controls_active_ne1').orderBy('match_id', 'rownum')
display(controls_1)

# COMMAND ----------

# checks
count_varlist(cases_1, ['person_id', 'index_date'])
count_varlist(controls_1, ['control_id'])

# COMMAND ----------

# Join the cases and with potential controls on a 1:n basis by match variables (controls ordered by a stable random row number)

join_1 = (
  cases_1.orderBy('match_id', 'rownum')
  .join(controls_1.orderBy('match_id', 'rownum'), ['match_id','rownum'], how='left')
  .withColumn('CONTROL_FLAG', f.when(f.col('control_id').isNotNull(), 1).otherwise(0))
  .withColumn('CONTROL_FLAG_INVALID',
               f.when(
                 (f.col('control_id').isNotNull()) 
                 & (f.col('date_of_death').isNotNull()) 
                 & (f.col('date_of_death') <= f.col('index_date')), 1
                 )
               .when(
                 (f.col('control_id').isNotNull()) 
                 & (f.col('diab_date').isNotNull()) 
                 & (f.col('diab_date') <= f.col('index_date')), 1
                 )
               .when(
                 (f.col('control_id').isNotNull()) 
                 & (f.col('covid_first_date').isNotNull()) 
                 & (f.col('covid_first_date') <= f.col('index_date')), 1
                 )
               .when(
                 (f.col('control_id').isNull()), None
                 )
               .otherwise(0))
  .select('person_id','match_id','rownum','control_no','control_id','index_date',
           'date_of_death','diab_date','covid_first_date','control_rand','CONTROL_FLAG','CONTROL_FLAG_INVALID')
  .orderBy('match_id', 'control_no')
)
save_table(df=join_1, out_name=f'{proj}_join_e1', save_previous=False)

join_1 = spark.table(f'{dsa}.{proj}_join_e1').orderBy('match_id', 'control_no')
tmpt = tab(join_1, 'CONTROL_FLAG'); print()
tmpt = tab(join_1, 'CONTROL_FLAG_INVALID'); print()
tmpt = tab(join_1, 'CONTROL_FLAG', 'CONTROL_FLAG_INVALID'); print()
display(join_1)

# COMMAND ----------

# save cases that matched
cases_matched = (
  join_1
  .where((f.col('CONTROL_FLAG') == 1) & (f.col('CONTROL_FLAG_INVALID') == 0))
  .select('person_id','match_id','control_no','control_id','index_date',
           'date_of_death','diab_date','covid_first_date')
  .orderBy('match_id','person_id', 'control_no')
)
count_varlist(cases_matched, ['person_id', 'index_date'])
display(cases_matched.where((f.col('person_id') == f.col('control_id'))))

# COMMAND ----------

# MAGIC %md
# MAGIC # 4 Check

# COMMAND ----------

win_controls = Window.partitionBy('person_id').orderBy('control_no')
cases_matched = (cases_matched
                   .withColumn('control_no', f.row_number().over(win_controls))
)
display(cases_matched.select('person_id','match_id','control_no','control_id','index_date'))

# COMMAND ----------

controls_per_case_summary = (cases_matched
                             .groupBy('person_id')
                             .agg(f.sum(f.lit(1)).alias('n_controls'))
)
tab(controls_per_case_summary, 'n_controls')

# COMMAND ----------

final_matched = (cases_matched
                 .select('person_id','match_id','control_no','control_id','index_date')
                 .unionByName(cases
                              .withColumn("control_no", f.lit(0))
                              .withColumn("control_id", f.lit(None))
                              .select('person_id','match_id','control_no','control_id','index_date'))
                 .withColumn("case_id", f.col('person_id'))
                 .withColumn('person_id', f.when(f.col('control_no')>0, f.col('control_id'))
                             .otherwise(f.col('person_id'))) 
                 .withColumn('exposed', f.when(f.col('control_no')==0, 1).otherwise(f.lit(0))    ) 
                 .select('person_id','exposed','case_id','control_id','control_no','index_date','match_id')
                 .orderBy('match_id','person_id', 'control_no')
)
count_varlist(final_matched, ['person_id'])
count_varlist(final_matched, ['control_id'])
count_varlist(final_matched, ['case_id'])
tab(final_matched, 'exposed')
tab(final_matched, 'control_no')
display(final_matched)

# COMMAND ----------

controls_per_case_summary = (final_matched
                             .groupBy('case_id')
                             .agg(f.sum(f.lit(1)).alias('n_controls'))
                             .withColumn('n_controls', f.col('n_controls') - f.lit(1))
)
tab(controls_per_case_summary, 'n_controls')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Save

# COMMAND ----------

save_table(df=final_matched, out_name=f'{proj}_final_matched', save_previous=False)
final_matched = spark.table(f'{dsa}.{proj}_final_matched')
count_varlist(final_matched, ['person_id', 'index_date'])

# COMMAND ----------

id1 = final_matched.select('person_id', 'index_date').filter(f.col('person_id').isNotNull()).distinct().count()
id1
id2 = final_matched.select('case_id').filter(f.col('person_id').isNotNull()).distinct().count()
id2
id3 = final_matched.select('control_id').filter(f.col('person_id').isNotNull()).distinct().count()
id3

# COMMAND ----------

controls_per_case_summary = (final_matched
                             .groupBy('case_id')
                             .agg(f.sum(f.lit(1)).alias('n_controls'))
                             .withColumn('n_controls', f.col('n_controls') - f.lit(1))
)
controls_per_case_summary = (controls_per_case_summary
                             .groupBy('n_controls')
                             .count()
                             .withColumn("n_controls",
                                         f.when(f.col("n_controls") == 0, "number of controls per case = 0")
                                         .when(f.col("n_controls") == 1, "number of controls per case = 1")
                                         .when(f.col("n_controls") == 2, "number of controls per case = 2")
                                         .when(f.col("n_controls") == 3, "number of controls per case = 3")
                                         .otherwise(f.col("n_controls"))
                             )
                             .withColumnRenamed('n_controls', 'step' )
)
display(controls_per_case_summary)

schema = t.StructType([t.StructField('step', t.StringType(), True),
                       t.StructField('count', t.StringType(), True)]
                       )
flow_chart = spark.createDataFrame([], schema)
step1 = [('total patients overall: ', id1)]
step2 = [('total cases: ', id2)]
step3 = [('total controls', id3)]

flow_chart = (flow_chart
              .union(spark.createDataFrame(step1, schema))
              .union(spark.createDataFrame(step2, schema))
              .union(spark.createDataFrame(step3, schema))
              .union(controls_per_case_summary)
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