from transforms.api import transform_df, Input, Output, configure
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import INPUTS, OUTPUTS
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, DateType, StructField, StringType, IntegerType, FloatType, DoubleType, NumericType, LongType

@configure(profile=["DRIVER_MEMORY_EXTRA_LARGE",
                    "EXECUTOR_MEMORY_LARGE",
                    "EXECUTOR_MEMORY_OVERHEAD_EXTRA_LARGE",
                    "NUM_EXECUTORS_32"])

@transform_df(
    Output(""),
    customize_concept_sets=Input(""),
    COHORT=Input(""),
    visits_of_interest=Input(""),
    COVID_deaths=Input(""),
    cohort_all_facts_table=Input(""),
    Sdoh_variables_all_patients=Input(""),
)

# def COVID_Patient_Summary_Table_LDS(cohort_all_facts_table, COHORT, visits_of_interest, COVID_deaths, customize_concept_sets, Sdoh_variables_all_patients):
def COVID_Patient_Summary_Table_LDS(cohort_all_facts_table, COHORT, visits_of_interest, COVID_deaths, customize_concept_sets):

    # SDOH_vars = Sdoh_variables_all_patients #dataset not joined in, just referenced here so that it will import with template for easy access if user wants to join to their summary tables

    visits_df = visits_of_interest
    deaths_df = COVID_deaths.select('person_id','COVID_patient_death')
    all_facts = cohort_all_facts_table
    fusion_sheet = customize_concept_sets

    # Step 1: Filter the DataFrame for qualitative measurements for pre and post
    pre_fusion = fusion_sheet.filter(
    (fusion_sheet["domain"] == "measurement") &
    (fusion_sheet["lower_bound"].contains("qual")) &
    (fusion_sheet["pre_during_post"].contains("pre"))
    )
    post_fusion = fusion_sheet.filter(
    (fusion_sheet["domain"] == "measurement") &
    (fusion_sheet["lower_bound"].contains("qual")) &
    (fusion_sheet["pre_during_post"].contains("post"))
    )
    
    # Step 2: Select the `indicator_prefix` column and collect it into a list
    pre_list = pre_fusion.select("indicator_prefix").distinct().rdd.flatMap(lambda x: x).collect()
    post_list = post_fusion.select("indicator_prefix").distinct().rdd.flatMap(lambda x: x).collect()

    # Step 3: Create a new list that includes the '_Pos' and '_Neg' versions of the indicator_prefix selected
    extended_pre_list = [f"{indicator}_Pos" for indicator in pre_list] + [f"{indicator}_Neg" for indicator in pre_list]
    extended_post_list = [f"{indicator}_Pos" for indicator in post_list] + [f"{indicator}_Neg" for indicator in post_list]
    
    pre_columns = list(
        fusion_sheet.filter(fusion_sheet.pre_during_post.contains('pre'))
        .select('indicator_prefix')
        .distinct().toPandas()['indicator_prefix'])
    pre_columns.extend(extended_pre_list + ['person_id', 'BMI_rounded', 'had_vaccine_administered'])
    during_columns = list(
        fusion_sheet.filter(fusion_sheet.pre_during_post.contains('during'))
        .select('indicator_prefix')
        .distinct().toPandas()['indicator_prefix'])
    during_columns.extend(['person_id', 'COVID_patient_death'])
    post_columns = list(
        fusion_sheet.filter(fusion_sheet.pre_during_post.contains('post'))
        .select('indicator_prefix')
        .distinct().toPandas()['indicator_prefix'])
    post_columns.extend(extended_post_list + ['person_id', 'BMI_rounded', 'is_first_reinfection', 'had_vaccine_administered'])
    
    df_pre_COVID = all_facts \
        .where(all_facts.pre_COVID==1) \
        .select(list(set(pre_columns) & set(all_facts.columns)))
    df_during_strong_COVID_hospitalization = all_facts \
        .where(all_facts.during_first_strong_COVID_hospitalization==1) \
        .select(list(set(during_columns) & set(all_facts.columns)))
    df_during_weak_COVID_hospitalization = all_facts \
        .where(all_facts.during_first_weak_COVID_hospitalization==1) \
        .select(list(set(during_columns) & set(all_facts.columns)))
    df_post_COVID = all_facts \
        .where(all_facts.post_COVID==1) \
        .select(list(set(post_columns) & set(all_facts.columns)))
    
    df_pre_COVID = df_pre_COVID.groupby('person_id').agg(
        F.max('BMI_rounded').alias('BMI_max_observed_or_calculated_before_or_day_of_covid'),
        *[F.max(col).alias(col + '_before_or_day_of_covid_indicator') for col in df_pre_COVID.columns if col not in ('person_id', 'BMI_rounded', 'had_vaccine_administered')],
        F.sum('had_vaccine_administered').alias('number_of_COVID_vaccine_doses_before_or_day_of_covid'))
    
    df_during_strong_COVID_hospitalization = df_during_strong_COVID_hospitalization.groupby('person_id').agg(
        *[F.max(col).alias(col + '_during_strong_covid_hospitalization_indicator') for col in df_during_strong_COVID_hospitalization.columns if col not in ('person_id')])

    df_during_weak_COVID_hospitalization = df_during_weak_COVID_hospitalization.groupby('person_id').agg(
        *[F.max(col).alias(col + '_during_weak_covid_hospitalization_indicator') for col in df_during_weak_COVID_hospitalization.columns if col not in ('person_id')])

    df_post_COVID = df_post_COVID.groupby('person_id').agg(
        F.max('BMI_rounded').alias('BMI_max_observed_or_calculated_post_covid'),
        *[F.max(col).alias(col + '_post_covid_indicator') for col in df_post_COVID.columns if col not in ('person_id', 'BMI_rounded', 'is_first_reinfection', 'had_vaccine_administered')],
        F.sum('had_vaccine_administered').alias('number_of_COVID_vaccine_doses_post_covid'),
        F.max('is_first_reinfection').alias('had_at_least_one_reinfection_post_covid_indicator'))

    #join above four tables on patient ID 
    df = df_pre_COVID.join(df_during_strong_COVID_hospitalization, 'person_id', 'outer')
    df = df.join(df_during_weak_COVID_hospitalization, 'person_id', 'outer')
    df = df.join(df_post_COVID, 'person_id', 'outer')
    
    df = df.join(visits_df,'person_id', 'outer')

    #already dependent on decision made in visits of interest node, no changes necessary here
    df = df.withColumn('strong_COVID_hospitalization_length_of_stay', 
        F.datediff("first_strong_COVID_hospitalization_end_date", "first_strong_COVID_hospitalization_start_date"))
    
    #join back in generic death flag for any patient with or without a date
    df = df.join(deaths_df, 'person_id', 'left').withColumnRenamed('COVID_patient_death', 'COVID_patient_death_indicator')
    #join back in death within fixed window post covid for patients with a date to use in severity of index infection
    df = df.join(all_facts.select('person_id','death_within_specified_window_post_covid').where(F.col('death_within_specified_window_post_covid')==1), 'person_id', 'left')
    #join in demographics and manifest data from cohort node
    df = COHORT.join(df, 'person_id','left')
    
    ## Capture list of Units for each Quantitative measurement
    unit_columns = [col for col in all_facts.columns if col.endswith('_unit')]
    # Loop through each '_unit' column and add a column with unique values as a list
    for col in unit_columns:
        # Collect unique values from the column
        unique_vals = all_facts.select(col).distinct().rdd.flatMap(lambda x: x).collect()
        # Add the unique values list as a new column at the end of df_a
        df = df.withColumn(f"{col}", F.array([F.lit(val) for val in unique_vals]))

    df = df.na.fill(value=0, subset = [col.name for col in df.schema.fields if not isinstance(col.dataType, DoubleType) | isinstance(col.dataType, LongType)])

    df = df.withColumn("Severity_Type", 
        F.when((df.COVID_first_PCR_or_AG_lab_positive.isNull() & df.COVID_first_diagnosis_date.isNull()), "No_COVID_index")
        .when((df.death_within_specified_window_post_covid == 1), "Death_within_n_days_after_COVID_index")
        .when((df.LL_ECMO_during_strong_covid_hospitalization_indicator == 1) | (df.LL_IMV_during_strong_covid_hospitalization_indicator == 1), "Severe_ECMO_IMV_in_Hosp_around_strong_signal_COVID_index")
        .when(df.first_strong_COVID_hospitalization_start_date.isNotNull(), "Moderate_Hosp_around_strong_signal_COVID_index")
        .when(df.first_strong_COVID_ED_only_start_date.isNotNull(), "Mild_ED_around_strong_signal_COVID_index")
        .when((df.LL_ECMO_during_weak_covid_hospitalization_indicator == 1) | (df.LL_IMV_during_weak_covid_hospitalization_indicator == 1), "Severe_ECMO_IMV_in_Hosp_around_weak_signal_COVID_index")
        .when(df.first_weak_COVID_hospitalization_start_date.isNotNull(), "Moderate_Hosp_around_weak_signal_COVID_index")
        .when(df.first_weak_COVID_ED_only_start_date.isNotNull(), "Mild_ED_around_weak_signal_COVID_index")
        .otherwise("Mild_No_ED_or_Hosp_around_COVID_index"))
    
    return df