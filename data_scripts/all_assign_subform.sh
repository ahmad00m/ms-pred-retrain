# NIST20 (commercial dataset)
ppm_diff=20
workers=64

# Write MAGMA-based subformulae
PYTHONPATH=. python data_scripts/forms/01_assign_subformulae.py \
  --data-dir data/spec_datasets/nist20 \
  --labels-file data/spec_datasets/nist20/labels.tsv \
  --use-magma \
  --mass-diff-thresh $ppm_diff \
  --output-dir-name magma_subform_50

# Write "no subformula" (use_all) baseline
PYTHONPATH=. python data_scripts/forms/01_assign_subformulae.py \
  --data-dir data/spec_datasets/nist20 \
  --labels-file data/spec_datasets/nist20/labels.tsv \
  --use-all \
  --output-dir-name no_subform

# Add intensities (03_add_form_intens.py)
PYTHONPATH=. python data_scripts/forms/03_add_form_intens.py \
  --num-workers $workers \
  --pred-form-folder data/spec_datasets/nist20/subformulae/magma_subform_50 \
  --true-form-folder data/spec_datasets/nist20/subformulae/no_subform \
  --add-raw \
  --binned-add \
  --out-form-folder data/spec_datasets/nist20/subformulae/magma_subform_50_with_raw
