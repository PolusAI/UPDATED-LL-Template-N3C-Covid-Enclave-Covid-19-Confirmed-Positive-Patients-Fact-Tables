

from transforms.api import transform_df, Input, Output, configure
import pyspark.sql.functions as F
from pyspark.sql.window import Window

@transform_df(
    Output(""),
    LL_concept_sets_fusion=Input(""),
    LL_DO_NOT_DELETE_REQUIRED_concept_sets_confirmed=Input(""),
    concept_set_members=Input(""),
    canonical_units_of_measure=Input(""),
)

def compute(LL_concept_sets_fusion, LL_DO_NOT_DELETE_REQUIRED_concept_sets_confirmed, canonical_units_of_measure, concept_set_members):

    # Bring in each dataset
    concepts_df = concept_set_members
    canonical_df = canonical_units_of_measure
    # Join the concepts_df to the LL_DO_NOT_DELETE dataset
    required = concepts_df.where((F.col('is_most_recent_version')=='true')) \
            .select('concept_set_name', 'codeset_id') \
            .dropDuplicates() \
            .join(LL_DO_NOT_DELETE_REQUIRED_concept_sets_confirmed.drop('codeset_id'), on = 'concept_set_name', how = 'right')
    # customizable = LL_concept_sets_fusion # SM Aug 20 2026: Modified code below, so the customizable dataset includes the codeset_id in the output
    customizable = LL_concept_sets_fusion.join(concepts_df.where((F.col('is_most_recent_version')=='true')).select('concept_set_name', 'codeset_id').dropDuplicates(), on = 'concept_set_name', how = 'left')
    # Join the LL_concept_sets dataset to the canonical units dataset
    measurements_df = customizable \
        .filter(customizable.domain.contains('measurement')) \
        .select('concept_set_name', 'indicator_prefix')
    
    ## Join Canonical units list with concept sets so each of the codeset IDS has a valid Concept name remove duplicates so each entry shows up once.
    codeset_df = canonical_df.join(concepts_df, on = 'codeset_id', how = 'left')
    count_df = codeset_df.groupBy('codeset_id').count()
    duplicate_values_df = count_df.filter(F.col('count') > 1).select('codeset_id')
    codeset_df = codeset_df.join(duplicate_values_df, 'codeset_id', 'left_anti')
    
    ## Join Required and customizable concept set lists. Modified by SM Aug 20 2026. This join will NOT work unless codeset_id is also in customizable - hence the join on line 32 above (to concept_set_members)
    # df = required.join(customizable, on = required.columns, how = 'outer') # codeset_id is NOT in customizable
    df = required.join(customizable, on = ['concept_set_name','indicator_prefix','domain','pre_during_post','codeset_id'], how = 'outer')

    ## Perform left_anti join to find rows in customized concepts in the measurement domain (the canonical measures) where concept_set_name does not exist in the codeset ids from canonical units
    missing_values_df = measurements_df.join(codeset_df, on = 'concept_set_name', how = 'left_anti')
    
    ## Find if any of the measurement concept sets do not have canonical units and are therefore unhamonized
    missing_count = missing_values_df.count()
    import warnings
    ## If there are concept sets without hamonized units, warn the user
    if missing_count > 0:
         warnings.warn("Some of the selected measurements have not undergone unit harmonization. We recomend exploring the units reported in quant_measurements_of_interest node")
    
    ## Create the list of concept sets to be included with a flag if they are measurements without harmonized units.
    df = df.join(missing_values_df.withColumnRenamed('indicator_prefix', 'missing_indicator_prefix'), on = 'concept_set_name', how = 'left') 
    df_with_flag = df.withColumn('harmonized_flag', F.when(F.col('missing_indicator_prefix').isNotNull(), 0).otherwise(1))
    df = df_with_flag.drop('missing_indicator_prefix')

    #### New step - added by SM Aug 20th 2026. 
    # This adds in the lower and upper bounds of lab measures that have been harmonized, and are in our fusion sheet
    # The upper and lower bounds are contained in the canonical_df dataset. This dataset however, does NOT contain the concept_set_name for each lab measure
    # Hence, we first attach the concept_set_name to each measure in the canonical_df (via the concept_set_members table). 
    # The updated canonical_df table now has the concept set name, measurement name, as well as upper_bound and lower bound
    canonical_df = canonical_units_of_measure.join(concept_set_members.select('codeset_id','concept_set_name'), on = 'codeset_id', how='inner').dropDuplicates()
    
    # Then join the updated canonical_df to our customize_concept_sets table (df) via the concept_set_name 
    # This will now ensure any lab measures we want to extract have the relevant lower_bound and upper_bound values
    # lower_bound and upper_bound are required in the downstream quant_measurements_of_interest script
    df = df.join(canonical_df.select('concept_set_name','min_acceptable_value','max_acceptable_value'), on = 'concept_set_name', how = 'left')
    df = df.withColumnRenamed('min_acceptable_value','lower_bound').withColumnRenamed('max_acceptable_value','upper_bound')
    
    # The final step is used to add the "is_qual" indicator to the COVID test concept sets in our table
    # This new column (is_qual) is used to filter these tests in the qual_measurements_of_interest script
    covid_concepts = ['ATLAS SARS-CoV-2 rt-PCR and AG','Atlas #818 [N3C] CovidAntibody retry','ResultPos','ResultNeg']
    covid_concepts_list = ','.join(["'{}'".format(x) for x in covid_concepts])
    df = df.withColumn('is_qual', F.expr('CASE WHEN concept_set_name IN ({}) THEN 1 ELSE 0 END'.format(covid_concepts_list)))
    
    return df
