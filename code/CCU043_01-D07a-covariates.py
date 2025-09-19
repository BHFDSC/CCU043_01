# Databricks notebook source
# MAGIC %md # CCU043_01-D07-covariates
# MAGIC  
# MAGIC **Description** This notebook obtains the covariate markers from GDPPR.
# MAGIC  
# MAGIC **Authors** Sharmin Shabnam, James Farrell (Health Data Science Team, BHF Data Science Centre)
# MAGIC  
# MAGIC **Reviewers** 
# MAGIC
# MAGIC **Acknowledgements** 
# MAGIC
# MAGIC **Data Output**
# MAGIC - **`ccu043_01_final_matched_cohort_w_covars`** : final matched cohort with covariates (BMI and smoking status) derived from GDPPR

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

# MAGIC %md # 0. Parameters

# COMMAND ----------

# MAGIC %run "./CCU043_01-D01-parameters"

# COMMAND ----------

def plot_hist(df, col, xlim1=None, xlim2=None):
  import matplotlib.pyplot as plt
  import seaborn as sns

  dfpd = df.select(col).toPandas()

  plt.figure(figsize=(10, 6))
  sns.histplot(dfpd[col], bins=20, kde=False, color="blue")
  plt.title(f"Distribution of {col}", fontsize=16)
  plt.xlabel(f"{col}", fontsize=14)
  plt.ylabel("Frequency", fontsize=14)
  plt.grid(True, linestyle='--', alpha=0.7)
  if xlim1 or xlim2:
    plt.xlim(xlim1, xlim2)
  plt.show()

# COMMAND ----------

# MAGIC %md # 1 Data

# COMMAND ----------

# Data tables
cohort = spark.table(f'{dsa}.{proj}_final_matched')
# count_var(cohort, 'person_id')

gdppr = extract_batch_from_archive(parameters_df_datasets, 'gdppr')
# count_var(gdppr, 'NHS_NUMBER_DEID')

# COMMAND ----------

# codelist
smok_dict = {'smoking_current':'1.Yes',
             'smoking_ex':'2.Ex',
             'smoking_never':'0.No'}
             
codelist_covariates_markers = spark.table(path_out_codelist_covariates)
codelist_covariates_markers = (codelist_covariates_markers
                                .replace(smok_dict,subset=['name'])
                               .dropDuplicates()
)
tmpt = tab(codelist_covariates_markers, 'name'); print()
display(codelist_covariates_markers)

# COMMAND ----------

# MAGIC %md # 2 Prepare

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 GDPPR

# COMMAND ----------

gdppr_prep = (
    gdppr
    .select(
        f.col('NHS_NUMBER_DEID').alias('person_id'),
        f.col('DATE').alias('date'),
        f.col('CODE').alias('code'),
        f.col('VALUE1_CONDITION').alias('value_1'),
        f.col('VALUE2_CONDITION').alias('value_2'),
    )
)
display(gdppr_prep)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Cohort
# MAGIC Censor dates:
# MAGIC - Max date: 1 day prior to operation
# MAGIC - Min date: 180 days prior to operation

# COMMAND ----------

cohort_prep = (
    cohort
    .select('person_id', 'index_date')
    .orderBy('person_id', 'index_date')
)
count_var(cohort_prep, 'person_id')
display(cohort_prep)

# COMMAND ----------

# MAGIC %md # 3 Create covariates

# COMMAND ----------

# MAGIC %md ## 3.1 Restrict to cohort IDs

# COMMAND ----------

# Run this below portion if there is any change in patient list
covariates_markers_gdppr = (
    merge(
        df1 = cohort_prep,
        df2 = gdppr_prep,
        ilist = ['person_id'],
        keep_results = ['both'], indicator = 0
    )
)
# display(covariates_markers_gdppr)
save_table(df=covariates_markers_gdppr, out_name=f'{proj}_tmp_gdppr_cohort', save_previous=False)
covariates_markers_gdppr = spark.table(f'{dsa}.{proj}_tmp_gdppr_cohort')
count_var(covariates_markers_gdppr, 'person_id')

# COMMAND ----------

# MAGIC %md ## 3.2 Restrict to codelist codes

# COMMAND ----------

# Run this below portion if there is any new codes added to the covriates code list
covariates_markers_codelist = (covariates_markers_gdppr
                               .join(codelist_covariates_markers, on='code', how='inner')
)
save_table(df=covariates_markers_codelist, out_name=f'{proj}_tmp_gdppr_codes', save_previous=False)
covariates_markers_codelist = spark.table(f'{dsa}.{proj}_tmp_gdppr_codes')
count_var(covariates_markers_codelist, 'person_id')
display(covariates_markers_codelist)

# COMMAND ----------

# MAGIC %md ## 3.3 BMI
# MAGIC - Restrict to date range
# MAGIC - Remove null values
# MAGIC - Create flag

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3.1 BMI Value

# COMMAND ----------

covariates_bmi_ori = (covariates_markers_codelist
                  .where((f.col('name') == 'bmi'))
                  .select('person_id','index_date','date','value_1')
                  .filter((f.col('value_1') >= 12)
                          & (f.col('value_1') <= 100))
                  .dropDuplicates()
                  .withColumnRenamed('value_1', 'bmi_value')
                  .withColumn("bmi_value", f.round(f.col("bmi_value"), 1))
                  .orderBy('person_id','index_date', 'date')
)
# count_var(covariates_bmi_ori, 'person_id')

window_spec = Window.partitionBy("person_id","index_date").orderBy("days_difference")
covariates_bmi = (covariates_bmi
                  .withColumn("days_difference",
                               f.abs(f.datediff(f.col("index_date"), f.col("date"))))
                  .filter(f.col('days_difference')<=2*365.25)
                  .withColumn("rank",f.row_number().over(window_spec))
                  .filter(f.col("rank") == 1).drop("rank").drop("days_difference")
                  .orderBy('person_id','index_date', 'date')
                  .withColumnRenamed('date', 'bmi_value_date')
)
count_var(covariates_bmi, 'person_id')
count_varlist(covariates_bmi, ['person_id', 'index_date'])
display(covariates_bmi)

# COMMAND ----------

# MAGIC %md ## 3.4 Smoking

# COMMAND ----------

covariates_smk = (covariates_markers_codelist
                  .where((f.col('name') == '0.No') 
                         | (f.col('name') == '1.Yes')
                         | (f.col('name') == '2.Ex'))
                  .withColumn("days_difference", f.datediff(f.col("index_date"), f.col("date")))
                  .withColumn("days_difference", f.abs(f.col("days_difference")))
                  .select('person_id','index_date','date','days_difference','name')
                  .withColumnRenamed('name', 'smoking_status')
                  .withColumnRenamed('date', 'smoking_date')
                  .filter(f.col('days_difference')<=2*365.25)
)
count_var(covariates_smk, 'person_id')
tmpt = tab(covariates_smk, 'smoking_status'); print()
display(covariates_smk)

# COMMAND ----------

window_spec = Window.partitionBy("person_id").orderBy("days_difference")
covariates_smk = (covariates_smk
                                .withColumn("rank",f.row_number().over(window_spec))
                                .filter(f.col("rank") == 1).drop("rank")
)
count_var(covariates_smk, 'person_id')
tmpt = tab(covariates_smk, 'smoking_status'); print()
display(covariates_smk)

# COMMAND ----------

# MAGIC %md # 4 Merge to first cohort

# COMMAND ----------

cohort = spark.table(f'{dsa}.{proj}_final_matched')
cohort = (cohort
          .join(covariates_bmi
                .select('person_id','index_date', 'bmi_value_date', 'bmi_value', 'bmi_cat_date', 'bmi_cat'),
                 on=['person_id','index_date'], how='left')
          .join(covariates_smk
                .select('person_id','index_date', 'smoking_date', 'smoking_status'),
                 on=['person_id','index_date'], how='left')
 )

# COMMAND ----------

save_table(df=cohort, out_name=f'{proj}_final_matched_cohort_w_covars', save_previous=False)
cohort = spark.table(f'{dsa}.{proj}_final_matched_cohort_w_covars')
count_varlist(cohort, ['person_id', 'index_date'])
display(cohort)

# COMMAND ----------

tmpt = tab(cohort, 'smoking_status'); print()

# COMMAND ----------

display(cohort.select('bmi_value').describe())