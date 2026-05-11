Here is the context to a specific GEO dataset that I am interested in harmonizing the {{ target_label }} labels for:

## GEO context
**Title**: {{ title }}
**Summary**: {{ summary }}
**Overall Design**: {{ overall_design }}

## Your Job
Based on the following table, for each value under the "Label" column, select the most relevant value in "Putative {{ ontology_name}} Labels".
{{ input }}

## Output Requirement
The output must match the following json_schema:
{{ json_schema }}