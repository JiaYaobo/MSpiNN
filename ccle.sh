python -u mspinn_ccle.py --k 2 --layer_sizes 100 50 --adam_learn_rate 1e-3\
                                    --n_epochs 200 --num_p 100\
                                    --seed 100\
                                    --lasso_param_ratio 0.1 --group_lasso_param 0.8 --ridge_param 0.01 >> logs/ccle/mspinn.csv