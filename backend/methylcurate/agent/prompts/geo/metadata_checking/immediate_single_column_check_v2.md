You provided the following erroneous resolution for the following concepts: {{ misformatted_concepts }}. We show this below:

{% if is_age %}
•  age
   •	You proposed the following resolution:
      •  {{ age_resolution }}
{% endif %}
{% if is_cell_type %}
•  cell_type
   •	You proposed the following resolution:
      •  {{ cell_type_resolution }}
{% endif %}
{% if is_disease_status %}
•  disease_status
   •	You proposed the following resolution:
      •  {{ disease_status_resolution }}
{% endif %}
{% if is_sex %}
•  sex
   •	You proposed the following resolution:
      •  {{ sex_resolution }}
{% endif %}
{% if is_subject_id %}
•  subject_id
   •	You proposed the following resolution:
      •  {{ subject_id_resolution }}
{% endif %}
{% if is_tissue %}
•  tissue
   •	You proposed the following resolution:
      •  {{ tissue_resolution }}
{% endif %}

In particular, you either did one of two things: you included the key_name in the regular expression pattern, which we expressly forbade; or you did not make your regex pattern generic (you hardcoded words into the pattern). You must change the pattern for each of {{ misformatted_concepts }}. Keep the following hard constraints in mind and check each resolution's notes for the specific changes you should make:

1) The regex MUST be compatible with Python’s built-in `re` module.
2) The regex MUST be SIMPLE:
   - NO lookbehind: do not use `(?<=...)` or `(?<!...)`
   - NO backreferences: do not use `\1`, `\g<1>`, etc.
   - NO conditionals, recursion, or named subroutines
   - NO inline flags like `(?i)`; flags are handled externally
   - Avoid heavy/ambiguous patterns: do not use `.*` unless it is the only way, and never use nested quantifiers
3) Capturing groups:
   - Use AT MOST ONE capturing group `(...)`.
   - Prefer ZERO capturing groups when possible.
   - If you use a group, it MUST capture the final value to extract.
   - Any other parentheses must be non-capturing `(?:...)` (but avoid even those if possible).
4) The pattern should match a single line of text and be robust to minor punctuation and whitespace.
5) The regex pattern should not include the key_name.
6) The regex pattern is used to extract a value, not validate it. Therefore, do not enumerate possible values and avoid alternation (|). Prefer a single capture group that matches the whole value using general tokens, e.g. (.+?) up to a delimiter, or ([^\n]+) for rest-of-line. If the extraction is already scoped to the exact substring, return (.+).