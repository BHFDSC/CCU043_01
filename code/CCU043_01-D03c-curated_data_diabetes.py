# Databricks notebook source
# MAGIC %md 
# MAGIC # CCU043_01-D03c-curated_data_diabetes
# MAGIC
# MAGIC **Description** This notebook extracts records for diabetes (any type) for each person.
# MAGIC
# MAGIC **Authors** Sharmin Shabnam
# MAGIC
# MAGIC **Created on** 2024.07.03
# MAGIC
# MAGIC **Last updated on** 2025.06.12
# MAGIC
# MAGIC **Data input** 
# MAGIC <br>functions - libraries - parameters
# MAGIC <br>Codelist outcomes
# MAGIC <br>ccu043_01_cur_hes_apc_all_years_long
# MAGIC <br>ccu043_01_cur_deaths_long
# MAGIC <br>gdppr
# MAGIC
# MAGIC **Data outputs**
# MAGIC - **`ccu043_01_cur_diabetes_all`**
# MAGIC
# MAGIC **Acknowledgements** Genevieve Cezard

# COMMAND ----------

spark.sql('CLEAR CACHE')

# COMMAND ----------

# DBTITLE 1,Libraries
import pyspark.sql.functions as f
from pyspark.sql.types import *
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

# MAGIC %md # 0 Parameters

# COMMAND ----------

# MAGIC %run "./CCU043_01-D01-parameters"

# COMMAND ----------

# MAGIC %md # 1 Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Diab Codes

# COMMAND ----------

codelist = spark.table(path_out_DM_algo_type)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 HES/Deaths

# COMMAND ----------

spark.sql(f"""REFRESH TABLE {path_cur_hes_apc_long}""")
hes_apc_long = spark.table(path_cur_hes_apc_long)
hes_apc_long = hes_apc_long.select([f.col(x).alias(x.lower()) for x in hes_apc_long.columns])

spark.sql(f"""REFRESH TABLE {path_cur_deaths_long}""")
deaths_long  = spark.table(path_cur_deaths_long)
deaths_long = deaths_long.select([f.col(x).alias(x.lower()) for x in deaths_long.columns])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 GDPPR

# COMMAND ----------

gdppr   = extract_batch_from_archive(parameters_df_datasets, 'gdppr')
gdppr = (gdppr
         .where((f.col('CODE').isNotNull()) | (f.col('CODE') != ''))
         .select(f.col('NHS_NUMBER_DEID').alias('PERSON_ID'),
                 f.coalesce('DATE','RECORD_DATE').alias('DATE'),'CODE')
)
gdppr = gdppr.select([f.col(x).alias(x.lower()) for x in gdppr.columns])

# COMMAND ----------

# MAGIC %md # 2 Prepare

# COMMAND ----------

tmpt = tab(codelist, 'terminology')

diabcode = codelist.filter((f.col('terminology')=="ICD-10")).select('name','code').distinct()
tmpt = tab(diabcode, 'name')

diabcode_sno = codelist.filter((f.col('terminology')=="SNOMED-CT")).select('name','code').distinct()
tmpt = tab(diabcode_sno, 'name')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 HES

# COMMAND ----------

_hes_apc = (hes_apc_long
            .join(f.broadcast(diabcode),'code')
            .select(['person_id', 'epistart','diag_position','name', 'code'])
            .withColumnRenamed('epistart', 'date')
)

tmpt = tab(_hes_apc,'code')
tmpt = tab(_hes_apc,'name')
count_var(_hes_apc, 'person_id')
null_counts = {col:_hes_apc.filter(_hes_apc[col].isNull()).count() for col in _hes_apc.columns}
null_counts

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 GDPPR

# COMMAND ----------

_gdppr = (gdppr
          .join(f.broadcast(diabcode_sno),'code')
          .select('person_id','date','code', 'name')
          .where(f.col('date').isNotNull())
)

tmpt = tab(_gdppr,'name')
count_var(_gdppr, 'person_id')
null_counts = {col:_gdppr.filter(_gdppr[col].isNull()).count() for col in _gdppr.columns}
null_counts

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Deaths

# COMMAND ----------

_death = (deaths_long
          .join(f.broadcast(diabcode),'code')
          .where(f.col('diag_position') == 'UNDERLYING')
          .select('person_id','date', 'code', 'name')
)

tmpt = tab(_death,'code')
tmpt = tab(_death,'name')
count_var(_death, 'person_id')
null_counts = {col:_death.filter(_death[col].isNull()).count() for col in _death.columns}
null_counts

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Combine

# COMMAND ----------

tmp1 = (_gdppr
        .withColumn('diag_position', f.lit(None))
        .withColumn('source', f.lit('1.gdppr'))
        .withColumn('code_type', f.lit('SNOMED'))
        .unionByName(_hes_apc.withColumn('source', f.lit('2.hes_apc'))
                     .withColumn('code_type',f.lit('ICD-10')))
        .unionByName(_death.withColumn('diag_position', f.lit('underlying'))
                        .withColumn('code_type',f.lit('ICD-10'))
                        .withColumn('source', f.lit('3.deaths')))
        .withColumnRenamed('code', 'diab_code')
        .withColumnRenamed('source', 'diab_source')
        .withColumnRenamed('name', 'diab_name')
        .withColumnRenamed('date', 'diab_date')
        .withColumnRenamed('diag_position', 'diab_diag_position')
        .withColumnRenamed('code_type', 'diab_code_type')
        )
tmpt = tab(tmp1,'diab_source')
tmpt = tab(tmp1,'diab_name')
count_var(tmp1, 'person_id')

# COMMAND ----------

a = tmp1.groupBy("diab_name").agg(f.countDistinct("person_id"))
a.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Save

# COMMAND ----------

save_table(df=tmp1, out_name=f'{proj}_cur_diabetes_all', save_previous=False)
tmp1 = spark.table(f'{dsa}.{proj}_cur_diabetes_all')

# COMMAND ----------

tmpt = tab(tmp1,'diab_source')
tmpt = tab(tmp1,'diab_name')
count_var(tmp1, 'person_id')

# COMMAND ----------

# MAGIC %md
# MAGIC #4 Check