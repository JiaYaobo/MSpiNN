for i in {1..10}; do
    python -u ensembles.py --balance 0.5 --err_dist 0 --n_train_obs 300 >> ../logs/ensembles/nonlinear_balance_300_ensembles_err_0.csv
    wait
done