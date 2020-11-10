dataset=${1}
root_dir=./data/${dataset}/

if [ ${dataset} = 'lastfm' ]; then
    python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 3 --dim 128
elif [ ${dataset} = 'wikipedia' ]; then
    python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 1 --dim 128
elif [ ${dataset} = 'reddit' ]; then
    python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 2 --dim 128 --k 1
elif [ ${dataset} = 'steam' ]; then
    python deepred/interaction_prediction.py --root-dir ${root_dir} --epoch 10 --dim 200
elif [ ${dataset} = 'sider' ]; then
    python deepred/interaction_prediction.py --root-dir ${root_dir} --epoch 10 --dim 128
elif [ ${dataset} = 'matador' ]; then
    python deepred/interaction_prediction.py --root-dir ${root_dir} --epoch 6 --dim 200
fi
