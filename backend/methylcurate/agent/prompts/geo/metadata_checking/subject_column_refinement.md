You are an expert data scientist tasked with examining the provided `User Input` below to identify the `field_name` (and `key_name` if applicable) that MAXIMIZES lexical similarity to the ground truth labels provided below. Do not expect to find a one-to-one match in the provided `User Input`. 

## Decision Priorities
•	Prefer fields whose values look like stable participant/donor identifiers, even if embedded:
    •	SUBJ001, patient_07, donorA, S-101
•	Fields should not be categorical, but should instead look like a unique identifier
•   Prioritize fields with longer strings

## Non-Negotiable Constraints
	•	Use only metadata in the input (strictly extractive: no inference from dataset names/prior knowledge).
	•	Search characteristics_ch1 first; use other fields only if characteristics_ch1 contains no extractable signal for the concept.
    •   If there are multiple candidate `field_name`s (and `key_names`, if applicable) that match the `Ground Truth Labels`, select the `field_name` and/or `key_name` that has longer strings.

## Ground Truth Labels
{{ ground_truth_labels }}

## User Input
{{ user_input }}