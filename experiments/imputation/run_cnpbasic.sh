#! /bin/bash
print_freq=25
epochs=251
model_name=npbasic

for run_number in 2 3 4 5 6 7 8 9; do
  python imputation.py --model_name ${model_name} --print_freq ${print_freq} \
  --run_number ${run_number} --epochs ${epochs}
done