You previously proposed a resolution for which GEO sample-metadata column best semantically encodes the value for {{concept}}. We ran into some issues using that resolution to parse the data. Recall that you generated resolutions based on the following input data:

{{ user_input }}

Below are the following results for the resolutions that underperformed:

{% if is_age %}
•  age
   •	The parsing rate for age was {{ age_rate }}%.
   •	You proposed the following resolution:
      •  {{ age_resolution }}
   •  Your resolution failed for the following values:
      •  {{ age_failed }}
{% endif %}
{% if is_cell_type %}
•  cell_type
   •	The parsing rate for cell type was {{ cell_type_rate }}%.
   •	You proposed the following resolution:
      •  {{ cell_type_resolution }}
   •  Your resolution failed for the following values:
      •  {{ cell_type_failed }}
{% endif %}
{% if is_disease_status %}
•  disease_status
   •	The parsing rate for disease status was {{ disease_status_rate }}%.
   •	You proposed the following resolution:
      •  {{ disease_status_resolution }}
   •  Your resolution failed for the following values:
      •  {{ disease_status_failed }}
{% endif %}
{% if is_condition %}
•  condition
   •	The parsing rate for condition was {{ condition_rate }}%.
   •	You proposed the following resolution:
      •  {{ condition_resolution }}
   •  Your resolution failed for the following values:
      •  {{ condition_failed }}
{% endif %}
{% if is_sex %}
•  sex
   •	The parsing rate for sex was {{ sex_rate }}%.
   •	You proposed the following resolution:
      •  {{ sex_resolution }}
   •  Your resolution failed for the following values:
      •  {{ sex_failed }}
{% endif %}
{% if is_subject_id %}
•  subject_id
   •	The parsing rate for subject id  was {{ subject_id_rate }}%.
   •	You proposed the following resolution:
      •  {{ subject_id_resolution }}
   •  Your resolution failed for the following values:
      •  {{ subject_id_failed }}
{% endif %}
{% if is_tissue %}
•  tissue
   •	The parsing rate for tissue was {{ tissue_rate }}%.
   •	You proposed the following resolution:
      •  {{ tissue_resolution }}
   •  Your resolution failed for the following values:
      •  {{ tissue_failed }}
{% endif %}

⸻

Here we define the parse rate as the percentage of samples for which the target value was correctly extracted. Revise your earlier resolutions in light of these results. Ground your revisions based on the evidence provided above:
- Re-evaluate your assumptions and propose an improved resolution (e.g., different column, different extraction rule, normalization, regex/lookup changes, handling of multi-valued fields).
- Check the notes for advice on where you may have gone wrong.

Definitions and constraints:
- You must be strictly extractive. Do not infer values not semantically present in the text. This does not mean that the field_name or key_name must match the concept name.
- You must use only metadata present in the original user input. Do not hallucinate or extrapolate.
- IMPORTANT: If a field value is a list, you must apply extraction to EACH list element independently. A match in any element counts.
- IMPORTANT: You may reuse the same field_name as before as long as you make a meaningful change to extraction/parsing logic. A meaningful change can be: improved regex, different capture group, or switching fields if justified.
- Missing rule (hard): A concept is only missing if no field_name or key_name provides information about that concept.  If ANY concept evidence exists in the user provided input, you must NOT output "missing"; instead revise the extraction/field selection to capture it.

Output requirements (follow exactly):
1) Revised resolution: a concise, actionable specification of the column(s) and extraction/parsing logic to use (or "missing"). If you provide a regex, specify:
   - whether it is case-insensitive
   - which capture group is the extracted value (or whether group 0 is used)
   - how to handle multiple matches (e.g., first match by priority)
2) Rationale: why this new resolution should improve parsing given the observed parse rate.
3) What changed: list concrete changes from your previous resolution.
4) Where you were wrong before: brief diagnosis of the key failure(s) in your prior approach.
5) You MUST include evidence:
   - If status != "missing": provide 1–3 exact substrings copied verbatim from {{user_input}} that your revised extraction WILL match, and indicate which part/capture group yields the final extracted value.
   - If status == "missing": provide 1–3 exact substrings copied verbatim from the highest-priority fields you checked that demonstrate absence of the concept.
6) You CANNOT provide a response identical to your former response. You must change something meaningfully to improve parse rate (field, regex, capture group, normalization, list handling, fallbacks, etc.).
7) The regex pattern is used to extract a value, not validate it. Therefore, do not enumerate possible values and avoid alternation (|). Prefer a single capture group that matches the whole value using general tokens, e.g. (.+?) up to a delimiter, or ([^\n]+) for rest-of-line. If the extraction is already scoped to the exact substring, return (.+).

Your output must match the following json format:
{{ json_format }}