# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC #CCU043_01-D02b-codelist_diabetes
# MAGIC
# MAGIC **Description** This notebook creates the codelist for 4 diabetes types and anti-diabetic medications to use in the diabetes algorithm.
# MAGIC
# MAGIC **Authors** Ewan Walker
# MAGIC
# MAGIC **Created on** 2023.11.08
# MAGIC
# MAGIC **Last updated on** 2024.01.31
# MAGIC
# MAGIC **Data input** functions - libraries - parameters
# MAGIC
# MAGIC **Data output** \
# MAGIC ccu043_01_out_codelist_dm_diabetes_algorithm \
# MAGIC ccu043_01_out_codelist_presc_diabetes_algorithm \
# MAGIC ccu043_01_out_codelist_careprocess_diabetes_algorithm
# MAGIC
# MAGIC **Reviewers** Genevieve Cezard, Sharmin Shabnam
# MAGIC
# MAGIC **Reviewed** 2025.06.01
# MAGIC
# MAGIC **Acknowledgements** 
# MAGIC
# MAGIC **Notes**

# COMMAND ----------

spark.sql('CLEAR CACHE')

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

# MAGIC %md # 1. Code lists

# COMMAND ----------

# MAGIC %md ## 1.1 Code list for Type 1 Diabetes Mellitus

# COMMAND ----------

# Diabetes type 1 (SNOMED / ICD10 codes only)
codelist_diabetes_type1 = spark.createDataFrame(
  [
    ('diabetes_type1','ICD-10','E10', 'Insulin-dependent diabetes mellitus','',''),
    ('diabetes_type1','ICD-10','O240','Diabetes mellitus in pregnancy: Pre-existing diabetes mellitus, insulin-dependent','',''),
    ('diabetes_type1','SNOMED-CT','23045005', 'Insulin dependent diabetes mellitus type IA','',''),
    ('diabetes_type1','SNOMED-CT', '28032008', 'Insulin dependent diabetes mellitus type IB','',''),
    ('diabetes_type1','SNOMED-CT', '46635009', 'Diabetes mellitus type 1','',''),
    ('diabetes_type1','SNOMED-CT', '190372001', 'Type I diabetes mellitus maturity onset','',''),
    ('diabetes_type1','SNOMED-CT', '199229001', 'Pre-existing type 1 diabetes mellitus','',''),
    ('diabetes_type1','SNOMED-CT', '290002008', 'Brittle type I diabetes mellitus','',''),
    ('diabetes_type1','SNOMED-CT', '313435000', 'Type I diabetes mellitus without complication','',''),
    ('diabetes_type1','SNOMED-CT', '444073006', 'Type I diabetes mellitus uncontrolled','',''),
    ('diabetes_type1','SNOMED-CT', '444074000', 'Type I diabetes mellitus well controlled','',''),
    ('diabetes_type1','SNOMED-CT', '472970003', 'History of diabetes mellitus type 1','',''),
    ('diabetes_type1','SNOMED-CT', '609564002', 'Pre-existing type 1 diabetes mellitus in pregnancy','',''),
    ('diabetes_type1','SNOMED-CT', '870528001', 'Newly diagnosed type 1 diabetes mellitus','',''),
    ('diabetes_type1','SNOMED-CT', '31321000119102', 'Diabetes mellitus type 1 without retinopathy','',''),
 ],
  ['name', 'terminology', 'code', 'term', 'code_type', 'RecordDate']  
)


# COMMAND ----------

display(codelist_diabetes_type1)

# COMMAND ----------

# MAGIC %md ## 1.2 Code list for Type 2 Diabetes Mellitus

# COMMAND ----------

# Diabetes type 2 (SNOMED / ICD10 codes only)
codelist_diabetes_type2 = spark.createDataFrame(
  [
    ('diabetes_type2','ICD-10','E11', 'Type 2 diabetes mellitus','',''),
    ('diabetes_type2','ICD-10','O241','Diabetes mellitus in pregnancy: Pre-existing diabetes mellitus, non-insulin-dependent','',''),
    ('diabetes_type2','SNOMED-CT','44054006', 'Diabetes mellitus type 2','',''),
    ('diabetes_type2','SNOMED-CT','81531005', 'Diabetes mellitus type 2 in obese','',''),
    ('diabetes_type2','SNOMED-CT','199230006','Pre-existing type 2 diabetes mellitus','',''),
    ('diabetes_type2','SNOMED-CT','237599002','Insulin treated type 2 diabetes mellitus','',''),
    ('diabetes_type2','SNOMED-CT','237627000','Pregnancy and type 2 diabetes mellitus','',''),
    ('diabetes_type2','SNOMED-CT','313436004','Type II diabetes mellitus without complication','',''),
    ('diabetes_type2','SNOMED-CT','314904008','Type II diabetes mellitus with neuropathic arthropathy','',''),
    ('diabetes_type2','SNOMED-CT','359642000','Diabetes mellitus type 2 in nonobese','',''),
    ('diabetes_type2','SNOMED-CT','443694000','Type II diabetes mellitus uncontrolled','',''),
    ('diabetes_type2','SNOMED-CT','444110003','Type II diabetes mellitus well controlled','',''),
    ('diabetes_type2','SNOMED-CT','445353002','Brittle type II diabetes mellitus','',''),
    ('diabetes_type2','SNOMED-CT','472969004','History of diabetes mellitus type 2','',''),
    ('diabetes_type2','SNOMED-CT','609567009','Pre-existing type 2 diabetes mellitus in pregnancy','',''),
    ('diabetes_type2','SNOMED-CT','703138006','Type II diabetes mellitus in remission','',''),
    ('diabetes_type2','SNOMED-CT','24471000000103','Type 2 diabetic on insulin','',''),
    ('diabetes_type2','SNOMED-CT','24481000000101','Type 2 diabetic on diet only','',''),
    ('diabetes_type2','SNOMED-CT','164971000119101','Type 2 diabetes mellitus controlled by diet','',''),
  ],
  ['name', 'terminology', 'code', 'term', 'code_type', 'RecordDate']  
)

# COMMAND ----------

display(codelist_diabetes_type2)

# COMMAND ----------

# MAGIC %md ## 1.3 Code list for Gestational Diabetes Mellitus

# COMMAND ----------

# Gestational Diabetes (SNOMED / ICD10 codes only)
codelist_diabetes_gestational = spark.createDataFrame(
  [
    ('diabetes_gestational','ICD-10','O244','Diabetes mellitus in pregnancy: Diabetes mellitus arising in pregnancy','',''),
    ('diabetes_gestational','ICD-10','O249','Diabetes mellitus in pregnancy: Diabetes mellitus in pregnancy, unspecified','',''),
    ('diabetes_gestational','SNOMED-CT','46894009','Gestational diabetes mellitus, class A>2< (disorder)','',''),
    ('diabetes_gestational','SNOMED-CT','11687002','Gestational diabetes mellitus (disorder)','',''),
    ('diabetes_gestational','SNOMED-CT','199223000','Diabetes mellitus during pregnancy, childbirth and the puerperium','',''),
    ('diabetes_gestational','SNOMED-CT','71546005','Gestational diabetes mellitus, class B>1< (disorder)','',''),
    ('diabetes_gestational','SNOMED-CT','75022004','Gestational diabetes mellitus, class A>1< (disorder)','',''),
    ('diabetes_gestational','SNOMED-CT','420491007','Gestational diabetes mellitus, class H (disorder)','',''),
    ('diabetes_gestational','SNOMED-CT','420738003','Gestational diabetes mellitus, class T (disorder)','',''),
    ('diabetes_gestational','SNOMED-CT','420989005','Gestational diabetes mellitus, class R (disorder)','',''),
    ('diabetes_gestational','SNOMED-CT','421223006','Gestational diabetes mellitus, class F (disorder)','',''),
    ('diabetes_gestational','SNOMED-CT','421389009','Gestational diabetes mellitus, class C (disorder)','',''),
    ('diabetes_gestational','SNOMED-CT','421443003','Gestational diabetes mellitus, class D (disorder)','',''),
    ('diabetes_gestational','SNOMED-CT','422155003','Gestational diabetes mellitus, class B (disorder)','',''),
    ('diabetes_gestational','SNOMED-CT','40801000119106','Gestational diabetes mellitus complicating pregnancy (disorder)','',''),
    ('diabetes_gestational','SNOMED-CT','10753491000119100','Gestational diabetes mellitus in childbirth (disorder)','',''),
  ],
  ['name', 'terminology', 'code', 'term', 'code_type', 'RecordDate']  
)


# COMMAND ----------

display(codelist_diabetes_gestational)

# COMMAND ----------

# MAGIC %md ## 1.4 Code list for Other/Non-specific/Generic Diabetes Mellitus

# COMMAND ----------

# Other/Non-specific/Generic Diabetes (SNOMED / ICD10 codes only)
codelist_diabetes_other = spark.createDataFrame(
  [
	('diabetes_other','ICD-10','E13','Other specified diabetes mellitus','',''),
 	('diabetes_other','ICD-10','E14', 'Unspecified diabetes mellitus','',''),
    ('diabetes_other','ICD-10','O243','Diabetes mellitus in pregnancy: Pre-existing diabetes mellitus, unspecified','',''),
    ('diabetes_other','SNOMED-CT','2751001','Fibrocalculous pancreatic diabetes','',''),
	('diabetes_other','SNOMED-CT','4855003','Retinopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','5368009','Drug-induced diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','5969009','Diabetes mellitus associated with genetic syndrome','',''),
	('diabetes_other','SNOMED-CT','8801005','Secondary diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','11530004','Brittle diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','19378003','Pseudotabes due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','20678000','Extreme insulin resistance with acanthosis nigricans, hirsutism AND autoantibodies to the insulin receptors','',''),
	('diabetes_other','SNOMED-CT','24203005','Extreme insulin resistance with acanthosis nigricans, hirsutism AND abnormal insulin receptors','',''),
	('diabetes_other','SNOMED-CT','25093002','Disorder of eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','25412000','Microaneurysm of retinal artery due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','25907005','Diabetic gangrene','',''),
	('diabetes_other','SNOMED-CT','26298008','Ketoacidotic coma due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','33559001','Pineal hyperplasia AND diabetes mellitus syndrome','',''),
	('diabetes_other','SNOMED-CT','35777006','Mononeuropathy multiplex due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','38046004','Diffuse glomerulosclerosis of kidney due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','38205001','Diarrhea due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','39058009','Lumbosacral radiculoplexus neuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','39127005','Symmetric proximal motor neuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','39181008','Radiculoplexus neuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','42954008','Diabetes mellitus associated with receptor abnormality','',''),
	('diabetes_other','SNOMED-CT','43959009','Cataract of eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','48951005','Bullous disease due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','49455004','Polyneuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','50620007','Autonomic neuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','51002006','Diabetes mellitus associated with pancreatic disease','',''),
	('diabetes_other','SNOMED-CT','54181000','Diabetes-nephrosis syndrome','',''),
	('diabetes_other','SNOMED-CT','57886004','Protein-deficient diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','59079001','Diabetes mellitus associated with hormonal etiology','',''),
	('diabetes_other','SNOMED-CT','59276001','Proliferative retinopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','62260007','Pretibial pigmental patches due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','63510008','Nodular glomerulosclerosis of kidney due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','70694009','Diabetes mellitus AND insipidus with optic atrophy AND deafness','',''),
	('diabetes_other','SNOMED-CT','73211009','Diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','74627003','Complication due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','75524006','Malnutrition related diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','75682002','Diabetes mellitus caused by insulin receptor antibodies','',''),
 	('diabetes_other','SNOMED-CT','79554005','Asymmetric proximal motor neuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','80660001','Mauriacs syndrome','',''),
	('diabetes_other','SNOMED-CT','81830002','Mononeuropathy simplex due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','82980005','Anemia due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','91352004','Diabetes mellitus due to structurally abnormal insulin','',''),
	('diabetes_other','SNOMED-CT','111307005','Leprechaunism syndrome','',''),
	('diabetes_other','SNOMED-CT','111552007','Diabetes mellitus without complication','',''),
	('diabetes_other','SNOMED-CT','111556005','Ketoacidosis without coma due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','123763000','Houssays syndrome','',''),
	('diabetes_other','SNOMED-CT','126534007','Mixed sensorimotor polyneuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','126535008','Motor polyneuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','127011001','Sensory neuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','127012008','Lipoatrophic diabetes','',''),
	('diabetes_other','SNOMED-CT','127013003','Disorder of kidney due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','127014009','Peripheral angiopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','170745003','Diabetic on diet only','',''),
	('diabetes_other','SNOMED-CT','170747006','Diabetic on insulin','',''),
	('diabetes_other','SNOMED-CT','170763003','Diabetic - good control','',''),
	('diabetes_other','SNOMED-CT','170766006','Loss of hypoglycemic warning due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','190329007','Diabetes mellitus with hyperosmolar coma','',''),
	('diabetes_other','SNOMED-CT','190406000','Ketoacidosis due to malnutrition related diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','190407009','Malnutrition-related diabetes mellitus with renal complications','',''),
	('diabetes_other','SNOMED-CT','190410002','Malnutrition-related diabetes mellitus with peripheral circulatory complications','',''),
	('diabetes_other','SNOMED-CT','190411003','Multiple complications due to malnutrition related diabetes','',''),
	('diabetes_other','SNOMED-CT','190412005','Malnutrition-related diabetes mellitus without complications','',''),
	('diabetes_other','SNOMED-CT','190416008','Steroid-induced diabetes mellitus without complication','',''),
	('diabetes_other','SNOMED-CT','190447002','Steroid-induced diabetes','',''),
	('diabetes_other','SNOMED-CT','193141005','Mononeuritis multiplex due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','193183000','Acute painful neuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','193184006','Chronic painful neuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','193185007','Asymptomatic neuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','193349004','Preproliferative retinopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','193350004','Advanced maculopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','193489006','Iritis due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','197605007','Nephrotic syndrome due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','199231005','Pre-existing malnutrition-related diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','200687002','Cellulitis of foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','201250006','Ischemic ulcer of foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','201251005','Neuropathic ulcer of foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','201252003','Mixed diabetic ulcer - foot','',''),
	('diabetes_other','SNOMED-CT','201723002','Cheiroarthropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','201724008','Neuropathic arthropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','230572002','Neuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','230573007','Diabetic distal sensorimotor polyneuropathy','',''),
	('diabetes_other','SNOMED-CT','230574001','Acute painful polyneuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','230575000','Chronic painful polyneuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','230576004','Asymmetric polyneuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','230577008','Mononeuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','230578003','Diabetic truncal radiculopathy','',''),
	('diabetes_other','SNOMED-CT','230579006','Thoracic radiculopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','232020009','Disorder of macula due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','232021008','Proliferative retinopathy with optic disc neovascularization due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','232022001','Proliferative retinopathy with neovascularization elsewhere than the optic disc due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','232023006','Traction detachment of retina due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','236499007','Microalbuminuric nephropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','236500003','Proteinuric nephropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','237600004','Malnutrition-related diabetes mellitus - fibrocalculous','',''),
	('diabetes_other','SNOMED-CT','237601000','Secondary endocrine diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','237604008','Maturity onset diabetes of the young, type 2','',''),
	('diabetes_other','SNOMED-CT','237608006','Lipodystrophy, partial, with Rieger anomaly, short stature, and insulinopenic diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','237610008','Acrorenal field defect, ectodermal dysplasia, and lipoatrophic diabetes','',''),
	('diabetes_other','SNOMED-CT','237612000','Photomyoclonus, diabetes mellitus, deafness, nephropathy and cerebral dysfunction','',''),
	('diabetes_other','SNOMED-CT','237616002','Hypogonadism, diabetes mellitus, alopecia, mental retardation and electrocardiographic abnormalities','',''),
	('diabetes_other','SNOMED-CT','237617006','Megaloblastic anemia, thiamine-responsive, with diabetes mellitus and sensorineural deafness','',''),
	('diabetes_other','SNOMED-CT','237618001','Insulin-dependent diabetes mellitus secretory diarrhea syndrome','',''),
	('diabetes_other','SNOMED-CT','237619009','Diabetes-deafness syndrome maternally transmitted','',''),
	('diabetes_other','SNOMED-CT','237620003','Abnormal metabolic state due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','237621004','Severe hyperglycemia due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','237622006','Poor glycemic control','',''),
	('diabetes_other','SNOMED-CT','237632004','Hypoglycemic event due to diabetes','',''),
	('diabetes_other','SNOMED-CT','237633009','Hypoglycemia due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','237635002','Nocturnal hypoglycemia due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','237650006','Insulin resistance in diabetes','',''),
	('diabetes_other','SNOMED-CT','237652003','Insulin resistance - type B','',''),
	('diabetes_other','SNOMED-CT','238981002','Disorder of soft tissue due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','238982009','Dermopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','238983004','Thick skin syndrome due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','238984005','Rubeosis faciei due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','267467004','Diabetes mellitus (& [ketoacidosis])','',''),
	('diabetes_other','SNOMED-CT','267471001','Diabetes + eye manifestation (& [cataract] or [retinopathy])','',''),
	('diabetes_other','SNOMED-CT','267604001','Myasthenic syndrome due to diabetic mellitus','',''),
	('diabetes_other','SNOMED-CT','268519009','Diabetic - poor control','',''),
	('diabetes_other','SNOMED-CT','280137006','Diabetic foot','',''),
	('diabetes_other','SNOMED-CT','284449005','Congenital total lipodystrophy','',''),
	('diabetes_other','SNOMED-CT','308105005','On examination-Right diabetic foot at risk','',''),
	('diabetes_other','SNOMED-CT','308106006','On examination-Left diabetic foot at risk','',''),
	('diabetes_other','SNOMED-CT','309426007','Glomerulopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','310387003','Intracapillary glomerulosclerosis of kidney due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','310505005','Hyperosmolar non-ketotic state due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','311366001','Kimmelstiel-Wilson syndrome','',''),
	('diabetes_other','SNOMED-CT','311782002','Advanced retinal disease due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','312903003','Mild nonproliferative retinopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','312904009','Moderate nonproliferative retinopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','312905005','Severe nonproliferative retinopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','312906006','Non-high-risk proliferative retinopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','312907002','High risk proliferative retinopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','312908007','Quiescent proliferative retinopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','312909004','Proliferative retinopathy with iris neovascularization due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','312910009','Vitreous hemorrhage due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','312912001','Macular edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','314010006','Diffuse exudative maculopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','314011005','Focal exudative maculopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','314014002','Ischemic maculopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','314015001','Mixed maculopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','314537004','Optic papillopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','359611005','Neuropathy with neurologic complication due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','361216007','Femoral mononeuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','371087003','Ulcer of foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','390834004','Nonproliferative retinopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','398140007','Post hypoglycemic hyperglycemia due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399862001','High risk proliferative retinopathy without macular edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399863006','Very severe nonproliferative retinopathy without macular edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399864000','Macular edema not clinically significant due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399865004','Very severe proliferative retinopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399866003','Venous beading of retina due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399868002','Intraretinal microvascular anomaly due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399869005','High risk proliferative retinopathy not amenable to photocoagulation due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399870006','Non-high-risk proliferative retinopathy with no macular edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399871005','Visually threatening retinopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399872003','Severe nonproliferative retinopathy with clinically significant macular edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399873008','Severe nonproliferative retinopathy without macular edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399874002','High risk proliferative retinopathy with clinically significant macula edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399875001','Non-high-risk proliferative retinopathy with clinically significant macular edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399876000','Very severe nonproliferative retinopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','399877009','Very severe nonproliferative retinopathy with clinically significant macular edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','402864004','Wet gangrene of foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','408409007','On examination - right eye background diabetic retinopathy','',''),
	('diabetes_other','SNOMED-CT','408410002','On examination - left eye background diabetic retinopathy','',''),
	('diabetes_other','SNOMED-CT','408411003','On examination - right eye preproliferative diabetic retinopathy','',''),
	('diabetes_other','SNOMED-CT','408412005','On examination - left eye preproliferative diabetic retinopathy','',''),
	('diabetes_other','SNOMED-CT','408413000','On examination - right eye proliferative diabetic retinopathy','',''),
	('diabetes_other','SNOMED-CT','408414006','On examination - left eye proliferative diabetic retinopathy','',''),
	('diabetes_other','SNOMED-CT','408415007','On examination - right eye diabetic maculopathy','',''),
	('diabetes_other','SNOMED-CT','408416008','On examination - left eye diabetic maculopathy','',''),
	('diabetes_other','SNOMED-CT','408540003','Diabetes mellitus caused by non-steroid drugs','',''),
	('diabetes_other','SNOMED-CT','413183008','Diabetes mellitus caused by non-steroid drugs without complication','',''),
	('diabetes_other','SNOMED-CT','414894003','On examination - left eye stable treated proliferative diabetic retinopathy','',''),
	('diabetes_other','SNOMED-CT','414910007','On examination - right eye stable treated proliferative diabetic retinopathy','',''),
	('diabetes_other','SNOMED-CT','417677008','On examination - sight threatening diabetic retinopathy','',''),
	('diabetes_other','SNOMED-CT','419100001','Infection of foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','420422005','Ketoacidosis due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','420662003','Coma due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','420683009','Disorder of nervous system due to malnutrition related diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','420996007','Coma due to malnutrition-related diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','421256007','Disorder of eye due to malnutrition related diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','421725003','Hypoglycemic coma due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','421895002','Peripheral vascular disorder due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','421966007','Non-ketotic non-hyperosmolar coma due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','422088007','Disorder of nervous system due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','422126006','Hyperosmolar coma due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','422183001','Skin ulcer due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','422275004','Gangrene due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','424736006','Peripheral neuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','425455002','Glomerulonephritis due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','426705001','Diabetes mellitus co-occurrent and due to cystic fibrosis','',''),
	('diabetes_other','SNOMED-CT','426875007','Latent autoimmune diabetes mellitus in adult','',''),
	('diabetes_other','SNOMED-CT','427089005','Diabetes mellitus due to cystic fibrosis','',''),
	('diabetes_other','SNOMED-CT','427943001','Ophthalmoplegia due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','441628001','Multiple complications due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','441656006','Hyperglycemic crisis due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','445170001','Macroalbuminuric nephropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','445260006','Posttransplant diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','472972006','History of maturity onset diabetes mellitus in young','',''),
	('diabetes_other','SNOMED-CT','609561005','Maturity-onset diabetes of the young','',''),
	('diabetes_other','SNOMED-CT','609562003','Maturity onset diabetes of the young, type 1','',''),
	('diabetes_other','SNOMED-CT','609563008','Pre-existing diabetes mellitus in pregnancy','',''),
	('diabetes_other','SNOMED-CT','609565001','Permanent neonatal diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','609568004','Diabetes mellitus due to genetic defect in beta cell function','',''),
	('diabetes_other','SNOMED-CT','609569007','Diabetes mellitus due to genetic defect in insulin action','',''),
	('diabetes_other','SNOMED-CT','609570008','Maturity-onset diabetes of the young, type 3','',''),
	('diabetes_other','SNOMED-CT','609571007','Maturity-onset diabetes of the young, type 4','',''),
	('diabetes_other','SNOMED-CT','609572000','Maturity-onset diabetes of the young, type 5','',''),
	('diabetes_other','SNOMED-CT','609573005','Maturity-onset diabetes of the young, type 6','',''),
	('diabetes_other','SNOMED-CT','609574004','Maturity-onset diabetes of the young, type 7','',''),
	('diabetes_other','SNOMED-CT','609575003','Maturity-onset diabetes of the young, type 8','',''),
	('diabetes_other','SNOMED-CT','609576002','Maturity-onset diabetes of the young, type 9','',''),
	('diabetes_other','SNOMED-CT','609577006','Maturity-onset diabetes of the young, type 10','',''),
	('diabetes_other','SNOMED-CT','609578001','Maturity-onset diabetes of the young, type 11','',''),
	('diabetes_other','SNOMED-CT','703136005','Diabetes mellitus in remission','',''),
	('diabetes_other','SNOMED-CT','704241002','Fetal hypertrophic cardiomyopathy due to maternal diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','707221002','Glomerulosclerosis of kidney due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','709147009','Gingivitis co-occurrent with diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','713457002','Neovascular glaucoma due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','713704004','Gastroparesis due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','715439000','Familial partial lipodystrophy type 2','',''),
	('diabetes_other','SNOMED-CT','716362006','Gingival disease co-occurrent with diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','720519003','Atherosclerosis, deafness, diabetes, epilepsy, nephropathy syndrome','',''),
	('diabetes_other','SNOMED-CT','721973006','Lipodystrophy, intellectual disability, deafness syndrome','',''),
	('diabetes_other','SNOMED-CT','722206009','Pancreatic hypoplasia, diabetes mellitus, congenital heart disease syndrome','',''),
	('diabetes_other','SNOMED-CT','722454003','Intellectual disability, craniofacial dysmorphism, hypogonadism, diabetes mellitus syndrome','',''),
	('diabetes_other','SNOMED-CT','723074006','Renal papillary necrosis due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','724067006','Permanent neonatal diabetes mellitus with cerebellar agenesis syndrome','',''),
	('diabetes_other','SNOMED-CT','724136006','Mastopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','724810001','Radiculoplexoneuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','724876003','Lesion of skin due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','724997001','Lumbosacral plexopathy co-occurrent and due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','733072002','Alaninuria, microcephaly, dwarfism, enamel hypoplasia, diabetes mellitus syndrome','',''),
	('diabetes_other','SNOMED-CT','734022008','Wolfram-like syndrome','',''),
	('diabetes_other','SNOMED-CT','735199000','History of diabetes related lower limb amputation','',''),
	('diabetes_other','SNOMED-CT','735200002','Absence of lower limb due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','735537007','Hyperosmolar hyperglycemic coma due to diabetes mellitus without ketoacidosis','',''),
	('diabetes_other','SNOMED-CT','735538002','Lactic acidosis due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','735539005','Metabolic acidosis due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','737212004','Diabetes mellitus caused by chemical','',''),
	('diabetes_other','SNOMED-CT','762489000','Acute complication due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','763325000','Insulin resistance','',''),
	('diabetes_other','SNOMED-CT','768792007','Cataract of right eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','768793002','Cataract of left eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','768794008','Cataract of bilateral eyes due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','768797001','Iritis of right eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','768798006','Iritis of left eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','768799003','Iritis of bilateral eyes due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769181007','Preproliferative retinopathy of right eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769182000','Preproliferative retinopathy of left eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769183005','Mild nonproliferative retinopathy of right eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769184004','Mild nonproliferative retinopathy of left eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769185003','Moderate nonproliferative retinopathy of right eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769186002','Moderate nonproliferative retinopathy of left eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769187006','Severe nonproliferative retinopathy of right eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769188001','Severe nonproliferative retinopathy of left eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769190000','Very severe nonproliferative retinopathy of right eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769191001','Very severe nonproliferative retinopathy of left eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769217008','Macular edema of right eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769218003','Macular edema of left eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769219006','Macular edema due to type 1 diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769220000','Macular edema due to type 2 diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769221001','Clinically significant macular edema of right eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769222008','Clinically significant macular edema of left eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769244003','Disorder of right macula due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','769245002','Disorder of left macula due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770094004','Cervical radiculoplexus neuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770095003','Cranial nerve palsy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770096002','Erectile dysfunction due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770097006','Clinically significant macular edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770098001','Cranial nerve palsy due to type 1 diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770323005','Retinal edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770324004','Ischemia of retina due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770361008','Vitreous hemorrhage of right eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770362001','Vitreous hemorrhage of left eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770581008','Microaneurysm of right retinal artery due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770582001','Microaneurysm of left retinal artery due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770599000','Venous beading of right retina due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770600002','Venous beading of left retina due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770765001','Proliferative retinopathy of right eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','770766000','Proliferative retinopathy of left eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','782755007','Primary microcephaly, mild intellectual disability, young-onset diabetes syndrome','',''),
	('diabetes_other','SNOMED-CT','782825008','Primary microcephaly, epilepsy, permanent neonatal diabetes syndrome','',''),
	('diabetes_other','SNOMED-CT','783722008','Myopathy and diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','788878000','Cardiomyopathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','789562001','Ulcer of heel due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','789568002','Ulcer of midfoot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','789585000','Sensory polyneuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','816067005','Diabetes, hypogonadism, deafness, intellectual disability syndrome','',''),
	('diabetes_other','SNOMED-CT','816177009','Nonproliferative retinopathy of left eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','816178004','Nonproliferative retinopathy of right eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','816961009','Stable treated proliferative retinopathy of right eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','816962002','Stable treated proliferative retinopathy of left eye due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','822995009','Hyperglycemia due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','860712005','Disorder of macula of bilateral eyes due to diabetes mellitus present','',''),
	('diabetes_other','SNOMED-CT','860721006','Disorder of macula of bilateral eyes due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','860798008','Glaucoma due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','860883001','Generalized autonomic neuropathy due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','860977000','Ulcer of right foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','860978005','Ulcer of left foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','860979002','Chronic ulcer of right foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','860980004','Chronic ulcer of left foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','863879002','At low risk of ulcer of right foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','863880004','At low risk of ulcer of left foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','863881000','At moderate risk of ulcer of left foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','863882007','At risk of ulcer of left foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','863883002','At risk of ulcer of right foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','863884008','At high risk of ulcer of right foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','863885009','At moderate risk of ulcer of right foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','863886005','At high risk of ulcer of left foot due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','870420005','Severe nonproliferative retinopathy with venous beading of retina due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','870421009','Cystoid macular edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','870529009','Persistent macular edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','871778008','Centrally involved macular edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','871781003','Non centrally involved macular edema due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','103981000119101','Proliferative retinopathy following surgery due to diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','105401000119101','Diabetes mellitus due to pancreatic injury','',''),
	('diabetes_other','SNOMED-CT','106281000119103','Pre-existing diabetes mellitus in mother complicating childbirth','',''),
	('diabetes_other','SNOMED-CT','112991000000101','Lipoatrophic diabetes mellitus without complication','',''),
	('diabetes_other','SNOMED-CT','198121000000103','Hypoglycaemic warning impaired','',''),
	('diabetes_other','SNOMED-CT','198131000000101','Hypoglycaemic warning good','',''),
	('diabetes_other','SNOMED-CT','335621000000101','Maternally inherited diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','367261000119100','Hyperosmolarity due to drug induced diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','368171000119104','Dermatitis due to drug induced diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','368601000119102','Hyperosmolar coma due to secondary diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','368711000119106','Mild nonproliferative retinopathy due to secondary diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','368721000119104','Non-proliferative retinopathy due to secondary diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','368741000119105','Moderate non-proliferative retinopathy due to secondary diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','385041000000108','Diabetes mellitus with multiple complications','',''),
	('diabetes_other','SNOMED-CT','385051000000106','Pre-existing diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','532411000000102','Diabetes mellitus, adult onset, with no mention of complication','',''),
	('diabetes_other','SNOMED-CT','658011000000104','Diabetes mellitus with other specified manifestation','',''),
	('diabetes_other','SNOMED-CT','677741000119109','Non-diabetic proliferative retinopathy of bilateral eyes','',''),
	('diabetes_other','SNOMED-CT','771571000000102','History of secondary diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','773001000000103','Symptomatic diabetic peripheral neuropathy','',''),
	('diabetes_other','SNOMED-CT','775841000000109','Diabetic retinopathy detected by national screening programme','',''),
	('diabetes_other','SNOMED-CT','894741000000107','Hypoglycaemic warning absent','',''),
	('diabetes_other','SNOMED-CT','1102351000000100','Ketosis-prone diabetes mellitus','',''),
	('diabetes_other','SNOMED-CT','10754881000119100','Diabetes mellitus in mother complicating childbirth','',''),
	('diabetes_other','SNOMED-CT','530558861000132000','Atypical diabetes mellitus','',''),
	],
	['name','terminology', 'code', 'term', 'code_type', 'RecordDate']
)


# COMMAND ----------

display(codelist_diabetes_other)

# COMMAND ----------

# MAGIC %md # 2. Combine

# COMMAND ----------

codelists_dm = [codelist_diabetes_type1, codelist_diabetes_type2, codelist_diabetes_gestational, codelist_diabetes_other]

codelist_DM_diabetes_algorithm = reduce(DataFrame.unionAll, codelists_dm)

# COMMAND ----------

display(codelist_DM_diabetes_algorithm)

# COMMAND ----------

# MAGIC %md # 3. Check

# COMMAND ----------

#check Diabetes Mellitus Codes
tmpt = tab(codelist_DM_diabetes_algorithm, 'name', 'terminology')

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Add BHF DDSC Codes

# COMMAND ----------

diabetes_type1_path = '/Workspace/Repos/ss1279@leicester.ac.uk/ccu043/bhf_codelist/diabetes_type1.csv'
diabetes_type2_path = '/Workspace/Repos/ss1279@leicester.ac.uk/ccu043/bhf_codelist/diabetes_type2.csv'
diabetes_nos_path = '/Workspace/Repos/ss1279@leicester.ac.uk/ccu043/bhf_codelist/diabetes_nos.csv'
diabetes_other_path = '/Workspace/Repos/ss1279@leicester.ac.uk/ccu043/bhf_codelist/diabetes_other.csv'
diabetes_gest_path = '/Workspace/Repos/ss1279@leicester.ac.uk/ccu043/bhf_codelist/diabetes_removed.csv'
paths = [diabetes_type1_path, diabetes_type2_path, diabetes_nos_path, diabetes_other_path, diabetes_gest_path]

spark_dfs = []

for path in paths:
    pandas_df = pd.read_csv(path, keep_default_na=False)
    if path == diabetes_gest_path:
        #print(pandas_df.diabetes_type_level2.value_counts())
        pandas_df = pandas_df[pandas_df.diabetes_type_level2 == 'Gestational']
    spark_df = spark.createDataFrame(pandas_df)
    spark_dfs.append(spark_df)

if spark_dfs:
    diabetes_codelists_bhf = spark_dfs[0]
    for df in spark_dfs[1:]:
        diabetes_codelists_bhf = diabetes_codelists_bhf.union(df)

diabetes_codelists_bhf = (diabetes_codelists_bhf
                      .withColumn("diabetes_type",
                                         f.when(f.col("diabetes_type") == "Type 1", "diabetes_type1")
                                         .when(f.col("diabetes_type") == "Type 2", "diabetes_type2")
                                         .when(f.col("diabetes_type") == "Removed", "diabetes_gestational")
                                         .when(f.col("diabetes_type") == "Other", "diabetes_other")
                                         .when(f.col("diabetes_type") == "Diabetes NOS", "diabetes_other")
                                         .otherwise(f.col("diabetes_type"))
                             )
                      .withColumn("terminology",
                                         f.when(f.col("code").isin(['G590', 'G632', 'H280', 'H360', 'M142', 'N083', 'Y423']), "ICD-10")
                                         .when(f.col("terminology") == "SNOMED", "SNOMED-CT")
                                         .otherwise(f.col("terminology"))
                             )
                             .withColumnRenamed('code_description', 'term' )
                             .withColumnRenamed('diabetes_type', 'name' )
                             .select(['name', 'terminology', 'code', 'term'])
                             .filter(f.col('terminology') != "ICD10")
)
tmpt = tab(diabetes_codelists_bhf, 'name')
display(diabetes_codelists_bhf)


# COMMAND ----------

codelist_DM_diabetes_algorithm = (codelist_DM_diabetes_algorithm
                                  .select(['name', 'terminology', 'code', 'term'])
              .unionByName(diabetes_codelists_bhf)
              .dropDuplicates(['code'])
             )
display(codelist_DM_diabetes_algorithm)

# COMMAND ----------

tmpt = tab(codelist_DM_diabetes_algorithm, 'name', 'terminology')
a = codelist_DM_diabetes_algorithm.groupby(['code']).count().where('count > 1').sort('count',ascending=False)
display(a)

# COMMAND ----------

# MAGIC %md # 4. Save

# COMMAND ----------

save_table(df=codelist_DM_diabetes_algorithm, out_name=f'{proj}_out_codelist_DM_diabetes_algorithm', save_previous=False)