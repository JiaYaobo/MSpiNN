for i in {1..10}; do
    python -u ensembles.py --balance 0.5 --err_dist 1 --n_train_obs 300 >> logs/ensembles/nonlinear_balance_300_ensembles_err_1.csv
    wait
done

for i in {1..10}; do
    python -u ensembles.py --balance 0.5 --err_dist 2 --n_train_obs 300 >> logs/ensembles/nonlinear_balance_300_ensembles_err_2.csv
    wait
done

for i in {1..10}; do
    python -u ensembles.py --balance 0.5 --err_dist 0 --n_train_obs 500 >> logs/ensembles/nonlinear_balance_500_ensembles_err_0.csv
    wait
done

for i in {1..10}; do
    python -u ensembles.py --balance 0.5 --err_dist 1 --n_train_obs 500 >> logs/ensembles/nonlinear_balance_500_ensembles_err_1.csv
    wait
done

for i in {1..10}; do
    python -u ensembles.py --balance 0.5 --err_dist 2 --n_train_obs 500 >> logs/ensembles/nonlinear_balance_500_ensembles_err_2.csv
    wait
done

for i in {1..10}; do
    python -u ensembles.py --balance 0.7 --err_dist 0 --n_train_obs 300 >> logs/ensembles/nonlinear_imbalance_300_ensembles_err_0.csv
    wait
done

for i in {1..10}; do
    python -u ensembles.py --balance 0.7 --err_dist 1 --n_train_obs 300 >> logs/ensembles/nonlinear_imbalance_300_ensembles_err_1.csv
    wait
done

for i in {1..10}; do
    python -u ensembles.py --balance 0.7 --err_dist 2 --n_train_obs 300 >> logs/ensembles/nonlinear_imbalance_300_ensembles_err_2.csv
    wait
done

for i in {1..10}; do
    python -u ensembles.py --balance 0.7 --err_dist 0 --n_train_obs 500 >> logs/ensembles/nonlinear_imbalance_500_ensembles_err_0.csv
    wait
done

for i in {1..10}; do
    python -u ensembles.py --balance 0.7 --err_dist 1 --n_train_obs 500 >> logs/ensembles/nonlinear_imbalance_500_ensembles_err_1.csv
    wait
done

for i in {1..10}; do
    python -u ensembles.py --balance 0.7 --err_dist 2 --n_train_obs 500 >> logs/ensembles/nonlinear_imbalance_500_ensembles_err_2.csv
    wait
done