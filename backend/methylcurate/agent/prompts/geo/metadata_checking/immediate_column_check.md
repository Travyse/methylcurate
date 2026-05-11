You provided several erroneous resolutions listed below:

{{ erroneous_resolutions }}

In particular, you included the key_name in the regular expression pattern, which we expressly forbade. Please regenerate the offending resolutions with the following hard constraints in mind:

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
5) The regex pattern should not include key_name.
