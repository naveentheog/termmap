Put the HiLabs files here (they are not in this repo because the data is proprietary):
  hackathon_train.csv                                  (required: training data)
  icd.xlsx    or the original icd10cm_codes.xlsx       (optional: downloaded automatically if missing)
  snomed.xlsx or SNOMEDCT_US_codes_and_descriptions.xlsx (optional: downloaded automatically if missing)
Other locations: python predict.py --train <csv> --icd <xlsx> --snomed <xlsx> ...
