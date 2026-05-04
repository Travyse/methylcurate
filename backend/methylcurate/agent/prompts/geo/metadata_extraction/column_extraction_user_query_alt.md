You are producing a GEOMetadataExtractionResult.

Using the decision policy and the provided input data, try to extract the values corresponding to each concept of interest (which is represented by the keys of the resolutions attribute). 

## Dataset context
**Title**: {{ dataset_title }}
**Summary**: {{ dataset_summary }}
**Overall Design**: {{ dataset_overall_design }}

## Input Data

{{ user_input }}

## Special rules for characteristics_ch1

If field_name is "characteristics_ch1":
	•	The source strings are in "key: value" format.
	•	You must set key_name to the key of the pair you are extracting from.
	•	Your regex must extract the value only (not the key).
        Example: for age: 37, the extraction must return 37 (not age: 37).

## Output requirements
	•	artifact must be null.
	•	Always prefer characteristics_ch1 for disease_status and age.
	•	The regular expression must extract the concept value from the identified field
	•	If your internal reasoning (notes) identifies a field or sub-key (e.g., "age is in characteristics_ch1") that contains the concept, you must set status to resolved and identify a regex pattern to extract that value.
	•	Output must follow the following json schema:

{{json_schema}}