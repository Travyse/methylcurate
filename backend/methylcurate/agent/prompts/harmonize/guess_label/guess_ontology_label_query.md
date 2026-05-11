Here is the context to a specific GEO dataset that I am interested in harmonizing the {{ target_label }} labels for:

## GEO context
**Title**: {{ dataset_title }}
**Summary**: {{ dataset_summary }}
**Overall Design**: {{ dataset_overall_design }}
**Metadata Field Name**: {{ metadata_field_name }}
**Metadata Key Name (If Applicable)**: {{ metadata_field_key_name }}

## Your Job
We want you to guess, as closely as possible, the {{ ontology_name }} label for each of the following labels from the dataset described above. If any of the following labels are generic (such as "case"), refer to the GEO context (title, summary, overall design, metadata field name and key name) to infer what that generic label is referring to:

**{{ target_label }} Labels**: {{ labels }}

## Output Requirement
The output must match the following json_schema:
{{ json_schema }}