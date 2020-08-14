#! /bin/bash
print_freq=10
epochs=51
model_name=setofconduits

for run_number in 4 5 6 7 8 9; do
  python imputation.py --model_name ${model_name} --print_freq ${print_freq} \
  --run_number ${run_number} --epochs ${epochs}
done