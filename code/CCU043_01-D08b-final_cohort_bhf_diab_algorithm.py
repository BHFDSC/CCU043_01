# Databricks notebook source
# MAGIC %md # CCU043_01-D08a-final_cohort_bhf_diab_algorithm
# MAGIC  
# MAGIC **Description** This notebook creates the final cohort using BHF DDSC Diabetes algorithm and survival dataset with covariates, outcomes, and follow up times.
# MAGIC  
# MAGIC **Authors** Sharmin Shabnam 
# MAGIC
# MAGIC **Reviewers** 
# MAGIC
# MAGIC **Acknowledgements** Health Data Science Team, BHF Data Science Centre
# MAGIC
# MAGIC **Notes**
# MAGIC
# MAGIC **Data Output**
# MAGIC - **`ccu043_01_final_cohort_all_data_bhf_diab_outcome`** : final cohort with exposed non-exposed individuals, convariates, and outcomes
# MAGIC - **`ccu043_01_survival_cohort_bhf_diab_outcome`** : final dataset for survival analysis 

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

cohort = spark.table(f'{dsa}.{proj}_final_matched_cohort_w_covars')
count_var(cohort, 'person_id')
count_var(cohort, 'case_id')
count_var(cohort, 'control_id')
count_varlist(cohort, ['person_id', 'index_date'])
tmpt = tab(cohort, 'exposed', var2_unstyled=1); print()
display(cohort)

# COMMAND ----------

base_cohort = spark.table(f'{dsa}.{proj}_tmp_baseline_cohort_with_covid_diabetes')
count_var(base_cohort, 'person_id')

# COMMAND ----------

# MAGIC %md
# MAGIC # 2 Join Baseline Data

# COMMAND ----------

base_cohort = (base_cohort
               .select(['person_id',
                        'date_of_birth', 'sex', 'ethnicity_5_group', 'region', 'imd_quintile',
                        'date_of_death',
                        'diab_date','diab_name', 
                        'diab_date_bhf','diab_name_bhf', 
                        'covid_first_date','covid_first',
                        'covid_first_hosp', 'covid_first_hosp_date',
                        'study_end_date'])
                       )
        
cohort = (cohort
          .join(base_cohort, on='person_id', how='left')
)
count_varlist(cohort, ['person_id', 'index_date'])
tmpt = tab(cohort,'diab_name')
tmpt = tab(cohort,'diab_name_bhf')

cohort = (cohort
          .withColumn('diab_date_bhf', 
                      f.when(f.col('diab_name_bhf')=='Unlikely',
                              f.lit(None)).otherwise(f.col('diab_date_bhf')))
          .withColumn('diab_name_bhf', 
                      f.when(f.col('diab_name_bhf')=='Unlikely',
                              f.lit(None)).otherwise(f.col('diab_name_bhf')))
)
tmpt = tab(cohort,'diab_name_bhf')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.0 Remove those with prevalent diab from BHF algorithm

# COMMAND ----------

cohort = (cohort
    .where((f.col('diab_date_bhf').isNull()) | (f.col('diab_date_bhf') > f.col('index_date')))
    .drop('diab_date', 'diab_name', 'bmi_cat', 'bmi_cat_date')
)
count_varlist(cohort, ['person_id', 'index_date'])
tmpt = tab(cohort, 'exposed',  var2_unstyled=1); print()
tmpt = tab(cohort, 'diab_name_bhf',  var2_unstyled=1); print() 
tmpt = tab(cohort, 'diab_name_bhf','exposed',  var2_unstyled=1); print()

# COMMAND ----------

# MAGIC %md
# MAGIC ##2.1 Create Outcome Columns

# COMMAND ----------

cohort = (cohort
          .withColumn('age', f.round(f.datediff(f.col('index_date'), f.col('date_of_birth'))/365.25, 0))
          .withColumnRenamed('ethnicity_5_group', 'ethnicity')
          .withColumnRenamed('imd_quintile', 'imd')
          .withColumnRenamed('smoking_status', 'smoking_cat')
          .withColumn('covid_censor_date',
                   f.when(f.col('covid_first_date').isNull(), f.lit(None))
                   .when((f.col('index_date') == f.col('covid_first_date')), f.lit(None))
                   .otherwise(f.col('covid_first_date'))
                      )
          ### Diabetes type 1
          .withColumn('outcome_t1',
                  f.when((f.col('covid_censor_date').isNull()) 
                          & (f.col('diab_name_bhf') == 'diabetes_type1'),
                              f.lit(1)
                              )
                  .when((f.col('covid_censor_date').isNotNull()) 
                          & (f.col('diab_name_bhf') == 'diabetes_type1')
                          & (f.col('diab_date_bhf') <= f.col('covid_censor_date')),
                              f.lit(1)
                              )
                  .otherwise(f.lit(0))
                  )
          .withColumn("followup_end_t1",
                      f.when(f.col("outcome_t1") == 1, f.col("diab_date_bhf"))
                      .otherwise(f.least(f.col("date_of_death"), f.col("study_end_date"), f.col("covid_censor_date"))) 
                      )
          .withColumn( "followup_time_t1",f.datediff(f.col("followup_end_t1"), f.col("index_date")))
          .withColumn("followup_time_t1",
                      f.when(f.col("followup_time_t1") == 0, f.lit(0.5)).otherwise(f.col("followup_time_t1"))
                      )
          .withColumn("followup_time_t1", f.col("followup_time_t1") / 365.24)
          ### Diabetes type 2
          .withColumn('outcome_t2',
                  f.when((f.col('covid_censor_date').isNull()) 
                          & (f.col('diab_name_bhf') == 'diabetes_type2'),
                              f.lit(1)
                              )
                  .when((f.col('covid_censor_date').isNotNull()) 
                          & (f.col('diab_name_bhf') == 'diabetes_type2')
                          & (f.col('diab_date_bhf') <= f.col('covid_censor_date')),
                              f.lit(1)
                              )
                  .otherwise(f.lit(0))
                  )
          .withColumn("followup_end_t2",
                      f.when(f.col("outcome_t2") == 1, f.col("diab_date_bhf"))
                      .otherwise(f.least(f.col("date_of_death"), f.col("study_end_date"), f.col("covid_censor_date"))) 
                      )
          .withColumn("followup_time_t2",f.datediff(f.col("followup_end_t2"), f.col("index_date")))
          .withColumn("followup_time_t2",
                      f.when(f.col("followup_time_t2") == 0, f.lit(0.5)).otherwise(f.col("followup_time_t2"))
                      )
          .withColumn("followup_time_t2", f.col("followup_time_t2") / 365.24)
        )
display(cohort)

# COMMAND ----------

tmpt = tab(cohort, 'diab_name_bhf', var2_unstyled=1); print()
tmpt = tab(cohort, 'outcome_t1','exposed',  var2_unstyled=1); print()
tmpt = tab(cohort, 'outcome_t2','exposed',  var2_unstyled=1); print()

# COMMAND ----------

tmpt = tab(cohort, 'outcome_t1'); print()
tmpt = tab(cohort, 'outcome_t2'); print()
cohort.select('followup_time_t1').describe().show()
cohort.select('followup_time_t2').describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##2.2 Format Categorical Columns

# COMMAND ----------

cohort = cohort.withColumn(
    "death",
    f.when(f.col("date_of_death").isNotNull(), f.lit(1))
    .otherwise(f.lit(0))
)
cohort = (cohort
      .withColumn("bmi_cat",
    f.when(f.col("bmi_value") < 18.5, "0.Underweight")
    .when((f.col("bmi_value") >= 18.5) & (f.col("bmi_value") <= 24.9), "1.Normal")
    .when((f.col("bmi_value") >= 25) & (f.col("bmi_value") <= 29.9), "2.Overweight")
    .when((f.col("bmi_value") >= 30) & (f.col("bmi_value") <= 39.9), "3.Obesity")
    .when(f.col("bmi_value") >= 40, "4.Severe_obesity")
    .otherwise(f.lit('5.Unknown')))
)
cohort = cohort.withColumn(
    "age_cat",
    f.when((f.col("age") >= 18) & (f.col("age") <= 29), "0.18–29")
    .when((f.col("age") >= 30) & (f.col("age") <= 39), "1.30–39")
    .when((f.col("age") >= 40) & (f.col("age") <= 49), "2.40–49")
    .when((f.col("age") >= 50) & (f.col("age") <= 59), "3.50–59")
    .when((f.col("age") >= 60) & (f.col("age") <= 69), "4.60–69")
    .when((f.col("age") >= 70) & (f.col("age") <= 79), "5.70–79")
    .when(f.col("age") >= 80, "6.80+")
    .otherwise('Unknown') 
    )

tmpt = tab(cohort, 'bmi_cat', 'exposed', var2_unstyled=1); print()
tmpt = tab(cohort, 'age_cat', 'exposed', var2_unstyled=1); print()
tmpt = tab(cohort, 'death', 'exposed', var2_unstyled=1); print()

# COMMAND ----------

cohort = cohort.withColumn(
    "smoking_cat",
    f.when((f.col("smoking_cat").isNull()), "3.Unknown")
    .otherwise((f.col("smoking_cat")) 
    ))
cohort = cohort.withColumn(
    "ethnicity",
    f.when((f.col("ethnicity")=='Mixed or multiple ethnic groups'), "3.Mixed/Other")
    .when((f.col("ethnicity")=='Black, Black British, Caribbean or African'), "2.Black or Black British")
    .when((f.col("ethnicity")=='Asian or Asian British'), "1.Asian or Asian British")
    .when((f.col("ethnicity")=='Other ethnic group'), "3.Mixed/Other")
    .when((f.col("ethnicity")=='White'), "0.White")
    .when((f.col("ethnicity").isNull()), "4.Unknown")
    .otherwise((f.col("ethnicity")) 
    ))

tmpt = tab(cohort, 'smoking_cat', 'exposed', var2_unstyled=1); print()
tmpt = tab(cohort, 'ethnicity', 'exposed', var2_unstyled=1); print()

# COMMAND ----------

tmpt = tab(cohort, 'sex', 'exposed', var2_unstyled=1); print()
tmpt = tab(cohort, 'imd', 'exposed', var2_unstyled=1); print()
tmpt = tab(cohort, 'region', 'exposed', var2_unstyled=1); print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Save

# COMMAND ----------

save_table(df=cohort, out_name=f'{proj}_final_cohort_all_data_bhf_diab_outcome', save_previous=False)
cohort = spark.table(f'{dsa}.{proj}_final_cohort_all_data_bhf_diab_outcome')
count_varlist(cohort, ['person_id', 'index_date'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Save Surv Cohort

# COMMAND ----------

print(cohort.columns)
surv_cohort = (cohort
               .select('person_id', 'index_date', 'exposed',
                        'age',
                        'sex', 'ethnicity', 'region', 'imd', 'bmi_value',
                        'outcome_t1', 'outcome_t2', 
                        'followup_end_t1', 'followup_end_t2', 
                        'followup_time_t1', 'followup_time_t2')
                       )
display(surv_cohort)

# COMMAND ----------

save_table(df=surv_cohort, out_name=f'{proj}_survival_cohort_bhf_diab_outcome', save_previous=False)
surv_cohort = spark.table(f'{dsa}.{proj}_survival_cohort_bhf_diab_outcome')
count_varlist(surv_cohort, ['person_id', 'index_date'])

# COMMAND ----------

# MAGIC %md
# MAGIC # 3 Baseline Table

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.1 Categorical

# COMMAND ----------

categorical_vars_dict = {
    'age_cat':'Age (years), n (%)',
    'ethnicity':'Ethnicity, n (%)',
    'imd':'IMD (quintiles), n (%)',
    'region':'Region, n (%)',
    'bmi_cat':'BMI (Kg/m2), n (%)',
    'smoking_cat':'Smoking Status, n (%)',
    'outcome_t2':'Type 2, n (%)',
    'outcome_t1':'Type 1, n (%)'
    }
categorical_vars = list(categorical_vars_dict.keys())


summary_list = []
i = 0
for cat_var in categorical_vars:
    for exposed_value in [0, 1]:
        #gorical_summary = []
        for sex in ['All', 'M', 'F']:
            print(cat_var, exposed_value, sex)
            if sex == 'All':
                df_filtered = cohort.filter(f.col("exposed") == exposed_value)
            else:
                df_filtered = cohort.filter(f.col("exposed") == exposed_value).filter(f.col("sex") == sex)
            window_spec = Window.partitionBy()
            summary = (
                df_filtered
                .groupBy(cat_var)
                .agg(f.count("*").alias("count"))
                .withColumn("percentage",
                             f.round((f.col("count") / f.sum("count").over(window_spec)) * 100, 1))
                .toPandas()
            )
            summary["count"] = ((summary["count"] / 5).round() * 5).astype(int)
            summary = summary.rename(columns={"cat_var": "categories"})
            summary["variable"] = cat_var
            summary["sex"] = sex 
            summary["exposed"] = exposed_value 
            summary["order"] = i 
            summary = summary.rename(columns={cat_var: "category"})
            summary = summary.sort_values(by=['category'], ascending=True)
            summary_list.append(summary)
            #print(summary)
    i += 1

summary = pd.concat(summary_list, ignore_index=True)
summary = summary.astype(str)
summary = spark.createDataFrame(summary)
display(summary)

# COMMAND ----------

cat_summary = (summary
                 .withColumn("group", f.concat_ws(" ", f.col("exposed"), f.col("sex")))
                 .withColumn("count",
                              f.concat_ws(" ",
                                          f.col("count").cast("string"), 
                                          f.concat_ws("",
                                                      f.lit("("),
                                                       f.col("percentage").cast("string"),
                                                        f.lit(")")
                                                        )))
                 .select('category', 'count', 'group', 'order')
                 .groupBy('order',"category")
                 .pivot("group")
                 .agg({"count": "first"})
)

dict_list = []
i = 0
for cat_var in categorical_vars_dict:
    new_row = {col: ' ' for col in cat_summary.columns}  
    new_row["category"] = ' '+ categorical_vars_dict[cat_var]
    new_row["order"] = str(i)
    i += 1
    dict_list.append(new_row) 

df = spark.createDataFrame(dict_list)
window_spec = Window.orderBy('order',"category")
cat_summary = (cat_summary
                     .unionByName(df)
                     .orderBy('order',"category")
                     .withColumn("category", f.ltrim("category"))
    )
display(cat_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.2 Continuous

# COMMAND ----------

continuous_vars_dict = {
    'age':'Age (years)',
    'bmi_value':'BMI (Kg/m2)',
    'sbp':'Systolic',
    'dbp':'Diastolic',
    'followup_time_t1':'Follow up time (years) T1',
    'followup_time_t2':'Follow up time (years) T2'
    }
continuous_vars = list(continuous_vars_dict.keys())

summary_list = []
i = 1
for var in continuous_vars:
    for exposed_value in [0, 1]:
        for sex in ['All', 'M', 'F']:
            print(var, exposed_value, sex)
            if sex == 'All':
                df_filtered = cohort.filter(f.col("exposed") == exposed_value)
            else:
                df_filtered = cohort.filter(f.col("exposed") == exposed_value).filter(f.col("sex") == sex)
            agg_exprs=[
                    f.mean(f.col(var)).alias("mean"),
                    f.stddev(f.col(var)).alias("sd"),
                    f.expr(f"percentile_approx({var}, 0.5)").alias("median"),
                    f.expr(f"percentile_approx({var}, 0.25)").alias("p25"),
                    f.expr(f"percentile_approx({var}, 0.75)").alias("p75"),
                ]
            summary = df_filtered.agg(*agg_exprs).toPandas()
            summary["category"] = var
            summary["sex"] = sex 
            summary["exposed"] = exposed_value 
            summary["order"] = i 


            m = df_filtered.filter(f.col(var).isNull()).select(f.col(var)).count()
            t = df_filtered.select(f.col(var)).count()
            mp = (m / t) * 100 
            m = pd.DataFrame({"mean": [m],"sd": [mp],
                              'median': [''],'p25' : [''],  'p75': ['']})
            m["mean"] = ((m["mean"] / 5).round() * 5).astype(int)
            m["category"] = var+'_missing'
            m["sex"] = sex 
            m["exposed"] = exposed_value 
            m["order"] = i 

            summary = pd.concat([summary, m], ignore_index=True)
            print(summary)
            summary_list.append(summary)
    i += 1


summary_list2 = []
i = 0
for exposed_value in [0, 1]:
    t = cohort.filter(f.col("exposed") == exposed_value).select(f.col('person_id')).count()
    for sex in ['All', 'M', 'F']:
        print(exposed_value, sex)
        if sex == 'All':
            df_filtered = cohort.filter(f.col("exposed") == exposed_value)
        else:
            df_filtered = cohort.filter(f.col("exposed") == exposed_value).filter(f.col("sex") == sex)
        summary2 = df_filtered.agg(*agg_exprs).toPandas()
        m = df_filtered.select(f.col('person_id')).count()
        mp = (m / t) * 100 
        summary2 = pd.DataFrame({"mean": [m],"sd": [mp],
                              'median': [None],'p25' : [None],  'p75': [None]})
        summary2["mean"] = ((summary2["mean"] / 5).round() * 5).astype(int)
        summary2["category"] = 'N'
        summary2["sex"] = sex 
        summary2["exposed"] = exposed_value 
        summary2["order"] = i 
        summary_list.append(summary2)

summary = pd.concat(summary_list, ignore_index=True)
summary = summary.astype(str)
summary = spark.createDataFrame(summary)
display(summary)

# COMMAND ----------

cont_summary = (summary
                .withColumn("mean",
                            f.when(f.col("mean") % 1 == 0, f.col("mean").cast("int").cast("string"))
                            .otherwise(f.format_string("%.1f", f.round(f.col("mean"), 1))))
                 .withColumn("group", f.concat_ws(" ", f.col("exposed"), f.col("sex")))
                 .withColumn("mean_sd",
                              f.concat_ws(" ",
                                          f.col("mean"), 
                                          f.concat_ws("",
                                                      f.lit("("),
                                                       f.round(f.col("sd"), 1).cast("string"),
                                                        f.lit(")")
                                                        )))
                 .withColumn("median_iqr",
                              f.concat_ws(" ",
                                          f.round(f.col("median"), 1).cast("string"), 
                                          f.concat_ws("",
                                f.lit("["),f.round(f.col("p25"), 1).cast("string"),
                                f.lit(", "),f.round(f.col("p75"), 1).cast("string"),
                                                        f.lit("]")
                                                        )))
                .select('category','mean_sd', 'median_iqr', 'group', 'order')
)

df_mean = cont_summary.selectExpr("category", "'mean (SD)' as type", "mean_sd as count", "group", "order")
df_median = cont_summary.selectExpr("category", "'median [IQR]' as type", "median_iqr as count", "group", "order")

# Combine them using unionByName()
cont_summary = (df_mean
                .unionByName(df_median)
                .filter(f.col('count')!='[, ]').filter(f.col('count')!='0 (0.0)') 
                .orderBy('category',"type")
                .groupBy('order',"category", 'type')
                .pivot("group")
                .agg({'count': "first"})
                .replace(continuous_vars_dict, subset=["category"])
                .withColumn("type",
                            f.when((f.col("category") == "N") | (f.col("category").contains("missing")), "n (%)").otherwise(f.col("type")))
                .withColumn("category", f.concat_ws(", ", f.col("category"), f.col("type")))
                .drop('type')

)
display(cont_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Combine

# COMMAND ----------

df = cat_summary.unionByName(cont_summary).drop('order')
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #4 Incidence rate

# COMMAND ----------

df = cohort.select('*').withColumn(
    "age_band",
    f.when(f.col("age").between(18, 29), "18-29")
    .when(f.col("age").between(30, 39), "30-39")
    .when(f.col("age").between(40, 49), "40-49")
    .when(f.col("age").between(50, 59), "50-59")
    .when(f.col("age").between(60, 69), "60-69")
    .when(f.col("age").between(70, 79), "70-79")
    .otherwise("80+")
)

df_crude_ori = (df
            .groupBy("exposed", 'sex')
            .agg(
                f.sum("outcome_t1").alias("events_t1"),
                f.sum("followup_time_t1").alias("pyrs_t1"),
                f.sum("outcome_t2").alias("events_t2"),
                f.sum("followup_time_t2").alias("pyrs_t2")
                )
            .withColumn("crude_ir_t1", (f.col("events_t1") / f.col("pyrs_t1")) * 100000)
            .withColumn("crude_ir_t2", (f.col("events_t2") / f.col("pyrs_t2")) * 100000)
            .orderBy('exposed')
    )
display(df_crude_ori)

df_crude = (df
            .groupBy("exposed", 'sex',"age_band")
            .agg(
                f.sum("outcome_t1").alias("events_t1"),
                f.sum("followup_time_t1").alias("pyrs_t1"),
                f.sum("outcome_t2").alias("events_t2"),
                f.sum("followup_time_t2").alias("pyrs_t2")
                )
            .withColumn("crude_ir_t1", (f.col("events_t1") / f.col("pyrs_t1")) * 100000)
            .withColumn("crude_ir_t2", (f.col("events_t2") / f.col("pyrs_t2")) * 100000)
            .withColumn("ci_lower_t1",
                        f.col("crude_ir_t1") * f.exp(-1.96 / f.sqrt(f.col("events_t1"))))
            .withColumn("ci_upper_t1",
                        f.col("crude_ir_t1") * f.exp(1.96 / f.sqrt(f.col("events_t1"))))
            .withColumn( "ci_lower_t2",
                         f.col("crude_ir_t2") * f.exp(-1.96 / f.sqrt(f.col("events_t2"))))
            .withColumn("ci_upper_t2",
                        f.col("crude_ir_t2") * f.exp(1.96 / f.sqrt(f.col("events_t2"))))
            .orderBy('sex','age_band', 'exposed')
    )
#display(df_crude)

def get_standardised_rate(outcome):
    age_dist = (df
                .filter(f.col("exposed") == 0)
                .groupBy("age_band")
                .agg(f.sum(f"followup_time_{outcome}").alias("total_pyrs"))
                .withColumn("weight",
                            f.col("total_pyrs") / f.sum("total_pyrs").over(Window.partitionBy())
                            )
                .orderBy('age_band')
                
    )
    display(age_dist)

    df_st = (df_crude
             .join(age_dist, "age_band", "left")
             .withColumn(f"weighted_ir_{outcome}",
                          f.col(f"crude_ir_{outcome}") * f.col("weight"))
             .withColumn(f"var_std_ir_{outcome}",
                          (f.col("weight") * f.col(f"crude_ir_{outcome}")) ** 2 / f.col(f"events_{outcome}"))
             .groupBy("exposed", 'sex')
             .agg(f.sum(f"weighted_ir_{outcome}").alias(f"standardised_ir_{outcome}"),
                  f.sum(f"var_std_ir_{outcome}").alias(f"var_std_ir_{outcome}"))
             .withColumn(f"ci_lower_std_{outcome}",
                         f.col(f"standardised_ir_{outcome}") 
                         * f.exp(-1.96 / f.sqrt(f.col(f"var_std_ir_{outcome}"))))
             .withColumn(f"ci_upper_std_{outcome}",
                         f.col(f"standardised_ir_{outcome}") 
                         * f.exp(1.96 / f.sqrt(f.col(f"var_std_ir_{outcome}"))))   
             #.select("exposed", 'sex', f"standardised_ir_{outcome}")
    )                  
    return df_st

df_t1 = get_standardised_rate('t1')
#display(df_t1)

df_t2 = get_standardised_rate('t2')
#display(df_t2)

df_final = (df_t1
            .join(df_t2, ["exposed", 'sex'], "left")
            .join(df_crude_ori, ["exposed", 'sex'], "left")
            .select('sex', 'exposed', 'events_t1', 'pyrs_t1', 'events_t2', 'pyrs_t2', 'crude_ir_t1', 'crude_ir_t2', 'standardised_ir_t1', 'standardised_ir_t2')
            .orderBy('sex','exposed')
            .withColumn("crude_ir_t1", f.round("crude_ir_t1"))
            .withColumn("crude_ir_t2", f.round("crude_ir_t2"))
            .withColumn("standardised_ir_t1", f.round("standardised_ir_t1"))
            .withColumn("standardised_ir_t2", f.round("standardised_ir_t2"))
            .withColumn("events_t1", f.round(f.col("events_t1") / 5) * 5)
            .withColumn("events_t2", f.round(f.col("events_t2") / 5) * 5)
            .withColumn("pyrs_t1", f.round(f.col("pyrs_t1") / 5) * 5)
            .withColumn("pyrs_t2", f.round(f.col("pyrs_t2") / 5) * 5)
)
display(df_final)