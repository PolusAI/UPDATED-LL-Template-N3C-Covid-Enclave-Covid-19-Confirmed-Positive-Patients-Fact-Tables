from transforms.api import transform_df, Input, Output, configure
import pyspark.sql.functions as F
from pyspark.sql.window import Window

@transform_df(
    Output(""),
    measurement=Input(""),
    concept_set_members=Input(""),
    COHORT=Input(""),
    canonical_units_of_measure=Input(""),
    customize_concept_sets=Input(""),
)

def quant_measurements_of_interest(measurement, concept_set_members, COHORT, canonical_units_of_measure, customize_concept_sets):
    
    #bring in only cohort patient ids
    persons = COHORT.select('person_id')
    
    # #filter measurement table to only cohort patients # Edited Aug 21, 2026 by SM - moved this lower down and perform joins early on    
    # measurement_df = measurement \
    #     .select('person_id','measurement_date','measurement_concept_id','harmonized_value_as_number', 'harmonized_unit_concept_id', 'value_as_number', 'unit_concept_name') \
    #     .where(F.col('measurement_date').isNotNull()) \
    #     .where(F.col('harmonized_value_as_number').isNotNull() | F.col('value_as_number').isNotNull())  \
    #     .withColumnRenamed('measurement_date','date') \
    #     .withColumnRenamed('measurement_concept_id','concept_id') \
    #     .join(persons, 'person_id', 'inner') 

    # #filter fusion sheet for concept sets and their future variable names that have concepts in the measurements domain
    # fusion_df = customize_concept_sets \
    #     .filter(customize_concept_sets.domain.contains('measurement')) \
    #     .filter(customize_concept_sets["lower_bound"] != "qual") \
    #     .select('concept_set_name', 'codeset_id', 'indicator_prefix', 'lower_bound', 'upper_bound', 'harmonized_flag')

    # EDITED above BY SM AUG 20th 2026 - changed filtering condition to is_qual = 0
    fusion_df = customize_concept_sets \
        .filter(customize_concept_sets.domain.contains('measurement')) \
        .filter(customize_concept_sets["is_qual"] == 0) \
        .select('concept_set_name', 'codeset_id', 'indicator_prefix', 'lower_bound', 'upper_bound', 'harmonized_flag') 
     
    concept_members_df = concept_set_members
          
    codeset_id = False
    #filter concept set members table to only concept ids for the measurements of interest
    #prioritize codeset_id
    if codeset_id == True:
        #create concepts_df from codeset_id
        concepts_df_id = concept_members_df \
            .select('concept_id', 'codeset_id') \
            .join(fusion_df, 'codeset_id', 'inner') \
            .select('concept_id','indicator_prefix', 'lower_bound', 'upper_bound', 'harmonized_flag')

        #create concepts_df from concept_set_name
        concepts_df_name = concept_members_df \
            .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
            .where((F.col('is_most_recent_version')=='true')) \
            .join(fusion_df.where(F.col('codeset_id').isNull()), 'concept_set_name', 'inner') \
            .select('concept_id','indicator_prefix', 'lower_bound', 'upper_bound', 'harmonized_flag')

        concepts_df = concepts_df_name.join(concepts_df_id, ['concept_id', 'indicator_prefix', 'lower_bound', 'upper_bound', 'harmonized_flag'], 'outer')

    #use only concept_set_name. This will filter to measurement concept sets where is_qual=0 (i.e., continuous measures like height, weight)
    else:
        concepts_df = concept_members_df \
            .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
            .where(F.col('is_most_recent_version')=='true') \
            .join(fusion_df, 'concept_set_name', 'inner') \
            .select('concept_id','indicator_prefix', 'lower_bound', 'upper_bound', 'harmonized_flag')

    #pull units for harmonized measurements
    canonical_df = canonical_units_of_measure.select('omop_unit_concept_id', 'omop_unit_concept_name') \
        .withColumnRenamed('omop_unit_concept_id', 'harmonized_unit_concept_id') \
        .withColumnRenamed('omop_unit_concept_name', 'harmonized_unit')
    
    # #find measurements information based on matching concept ids for harmonized measurements of interest AND add units for the corresponding measurements
    # df = measurement_df.join(concepts_df, 'concept_id', 'inner') \
    #     .join(canonical_df, 'harmonized_unit_concept_id', 'left') \
    #     .drop('harmonized_unit_concept_id')

    #Edited by SM Aug 21, 2026. Adding broadcast to speed things up
    from pyspark.sql.functions import broadcast
    
    # Filter measurement table to only cohort patients - edited SM Aug 25 2026 (code moved here from the top)    
    df = measurement.withColumnRenamed('measurement_date','date') \
        .withColumnRenamed('measurement_concept_id','concept_id') \
        .where(F.col('date').isNotNull()) \
        .where(F.col('harmonized_value_as_number').isNotNull() | F.col('value_as_number').isNotNull())  \
        .join(broadcast(concepts_df), on = 'concept_id', how = 'inner') \
        .join(broadcast(canonical_df), on = 'harmonized_unit_concept_id', how = 'left') \
        .join(persons, 'person_id', 'inner') \
        .drop('harmonized_unit_concept_id') \
        .select('person_id',
        'date',
        'concept_id',
        'harmonized_value_as_number',
        'value_as_number',
        'unit_concept_name',
        'lower_bound',
        'upper_bound',
        'harmonized_flag',
        'indicator_prefix',
        'harmonized_unit')
        

    #filter harmonized values for those within upper and lower bounds set by unit_source_value
    #if non-harmonized value is used, record non-harminized unit
    df = df.withColumn('value', F.when(F.col('harmonized_value_as_number').between(F.col('lower_bound'), F.col('upper_bound')), F.col('harmonized_value_as_number'))
        .otherwise(0)) \
        .withColumn('value', F.when(F.col('harmonized_flag')==1, F.col('value'))
        .otherwise(F.col('value_as_number'))) \
        .withColumn('unit', F.when(F.col('harmonized_flag')==1, F.col('harmonized_unit'))
        .otherwise(F.col('unit_concept_name'))) \
        .select('person_id', 'date', 'indicator_prefix', 'value', 'unit') 
    
    #collapse to unique person and date and pivot on future variable name to create flag for rows associated with the concept sets for measurements of interest
    # # First, order the DataFrame by 'value' in descending order # EDITED SM AUG 21 2026: This line is apparently unnecessary
    # df = df.orderBy(F.desc('value'))

    # Then, select the first row per group, grouped by 'person_id', 'date', and 'indicator_prefix'
    df = (df.withColumn("row_num", F.row_number().over(Window.partitionBy("person_id", "date", "indicator_prefix").orderBy(F.desc("value")))).filter(F.col("row_num") == 1).drop("row_num"))

    # Now, pivot based on 'indicator_prefix' and aggregate using first() on 'value' and 'unit'
    df = df.persist() # Added by SM Aug 21, 2026 to enhance performance. 
    pivoted_df = (df.groupBy("person_id", "date").pivot("indicator_prefix").agg(F.first("value").alias("value"), F.first("unit").alias("unit")))
    
    cols = [col.replace('_value','').strip() for col in pivoted_df.columns]
    df = pivoted_df.toDF(*cols)
    
    #Find BMI closest to COVID using both reported/observed BMI and calculated BMI using height and weight.  Cutoffs for reasonable height, weight, and BMI are provided and can be changed by the template user.
    #add a calculated BMI for each visit date when height and weight available.  Note that if only one is available, it will result in zero
    #subsequent filter out rows that would have resulted from unreasonable calculated_BMI being used as best_BMI for the visit 
    BMI_df = df.withColumn('calculated_BMI', (F.col('WEIGHT')/(F.col('HEIGHT')*F.col('HEIGHT'))))
    
    BMI_df = BMI_df.withColumn('BMI_overall', F.when(F.col('BMI')>0, F.col('BMI')).otherwise(F.col('calculated_BMI'))) \
        .select('person_id','date','BMI_overall') \
        .withColumn('indicator_prefix', F.lit('BMI')) \
        .join(fusion_df, 'indicator_prefix', 'left')

    BMI_df = BMI_df.where(F.col('BMI_overall').between(F.col('lower_bound'), F.col('upper_bound'))) \
        .withColumn('BMI_rounded', F.round(F.col('BMI_overall'))) \
        .drop('BMI_overall')

    BMI_df = BMI_df.withColumn('OBESITY', F.when(F.col('BMI_rounded')>=30, 1).otherwise(0)) \
        .select('person_id', 'date', 'BMI_rounded', 'OBESITY')

    ## Check use case for where height and weight are in the optional concept sets    
    df = df.drop('BMI') \
        .join(BMI_df, ['person_id', 'date'], 'left')

    return df