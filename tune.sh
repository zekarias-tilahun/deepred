dataset=${1}
root_dir=./data/${dataset}/


if [ ${dataset} = 'lastfm' ] || [ ${dataset} = 'wikipedia' ] || [ ${dataset} = 'reddit' ] || [ ${dataset} = 'behance' ]; then
    python deepred/tune.py --root-dir ${root_dir} --nbr-size 20 100 --epochs 1 --trials 1 --temporal --log-level 3 --lr 0.0001 --dropout 0. 1. --reg-cof 0. 1. --min
else
    python deepred/tune.py --root-dir ${root_dir} --nbr-size 300 --epochs 1 --trials 50 --log-level 3 --lr 0. 1. --dropout 0. 1. --reg-cof 0. 1. --min
fi
