To evaluate the performance of the suggested resolutions for `beta_column` and `detection_column`, we extracted another random subset of data from the same dataset and applied your resolutions. The performance is described below:

## New Random Subset of Data

{{sample_data}}

## Performance 

### Beta Column Performance

- Your proposed regular expression: {{beta_pattern}}
- Columns identified as `beta_column`s based on your regular expression: {{beta_columns}}
- Columns identified as not being `beta_column`s based on your regular expression: {{not_beta_columns}}

### Detection Column Performance

- Your proposed regular expression: {{detection_pattern}}
- Columns identified as `detection_column`s based on your regular expression: {{detection_columns}}
- Columns identified as not being `detection_column`s based on your regular expression: {{not_detection_columns}}

## Your Task

Based on the above performance, check the following:

1. Are there columns identified as being `beta_column`s that shouldn't be identified as `beta_column`s based on how we defined what a `beta_column` is?
2. Are there columns identified as not being `beta_column`s that should be identified as `beta_column`s based on how we defined what a `beta_column` is?
3. Are there columns identified as being `detection_column`s that shouldn't be identified as `detection_column`s based on how we defined what a `detection_column` is?
4. Are there columns identified as not being `detection_column`s that should be identified as `detection_column`s based on how we defined what a `detection_column` is?

If you answer yes to any of those questions, suggest a new regular expression for `beta_column` and/or `detection_column` based on your findings. If you answer no to all of those questions, return your previous answer completely unchanged.

## Output Requirement

Return only valid json matching the following schema:
{{json_schema}}