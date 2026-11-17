# Local release validation

- 72 archived source/environment files match their recorded SHA-256.
- 65 Python source files were syntax-checked without importing or executing scientific code.
- Three selected weight hashes match the manuscript training configuration.
- Primary and discovery array headers were inspected without reading or changing values.
- 91 final-manuscript package files are unchanged.
- GitHub payloads contain no file above 25 MiB and no high-confidence token/private-key pattern was found.
- All 37,028 payload files across eight data archives passed streamed SHA-256 and ZIP CRC verification; the report accompanies the Zenodo files.
- No model, derivative, GRNBoost2, differential-expression or seed-sign analysis was run.
- No remote repository or Zenodo record was created; URLs and DOI remain unassigned.

Historical scientific code requires the platform-specific context described in
REPRODUCIBILITY.md. Syntax and integrity checks do not certify end-to-end replay.
