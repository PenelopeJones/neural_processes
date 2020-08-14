#! /bin/bash
print_freq=100
epochs=10001
model_name=npbasic
n_properties=159
dataname=Kinase


for run_number in 0; do
  python imputation.py --model_name ${model_name} --dataname ${dataname} --n_properties ${n_properties} \
  --print_freq ${print_freq} --run_number ${run_number} --epochs ${epochs}
done