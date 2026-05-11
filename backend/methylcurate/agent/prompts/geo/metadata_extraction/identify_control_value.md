You just finished exctracting all of the metadata from a NCBI Gene Expression Omnibus (GEO) dataset. You noticed that each sample has one of the following conditions: {{ disease_statuses }}; where we define condition as the grouping (case, control; treatment, no-treatment) or disease status (e.g. AD). The 'key_name' from which you extracted these statuses is: {{ key_name }}, which may or may not hold a hint. Which of these conditions, if any, do you believe is the healthy/control group value?

Your output must match the following json format:
{{ json_format }}