dataset=${1}
root_dir=./data/${dataset}/

if [ ${dataset} = 'lastfm' ]; then
    python deepred/deepred.py --root-dir ${root_dir} --epochs 10 --temporal --reg-cof 0.7 --lr .0001 --dropout .8 --nbr-size 100
elif [ ${dataset} = 'wikipedia' ]; then
    python deepred/deepred.py --root-dir ${root_dir} --epochs 10 --temporal --reg-cof 0.5 --lr .0001 --dropout .7 --nbr-size 50
elif [ ${dataset} = 'reddit' ]; then
    python deepred/deepred.py --root-dir ${root_dir} --epochs 10 --temporal --reg-cof .6 --lr .0001 --dropout .6 --nbr-size 100
elif [ ${dataset} = 'ctd' ]; then
    python deepred/deepred.py --root-dir ${root_dir} --epochs 10 --lr 0.0001 --dropout 0.8 --reg-cof 0.7 --nbr-size 200 --dim 128
elif [ ${dataset} = 'sider' ]; then
    python deepred/deepred.py --root-dir ${root_dir} --epochs 50 --lr 0.0001 --dropout 0.5 --reg-cof 0.6 --nbr-size 200 --dim 128
fi
