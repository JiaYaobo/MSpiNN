for i in {1..5}; do
    python -u mspinn_ccle.py --k 1 --layer_sizes 100 50  --adam_learn_rate 1e-3\
                                        --n_epochs 200 --num_p 100\
                                        --seed "$i"\
                                        --lasso_param_ratio 0.1 --group_lasso_param 0.1 --ridge_param 0.01 >> logs/ccle/spinn.csv
done