dataset=${1}
root_dir=./data/${dataset}/

if [ ${dataset} = 'lastfm' ]; then
#     python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 3
    for i in {1..10} ; do
      python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch $i
    done
elif [ ${dataset} = 'wikipedia' ]; then
#     python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 5
    for i in {1..10} ; do
      python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch $i
    done
elif [ ${dataset} = 'reddit' ]; then
    #python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 2
    for i in {1..10} ; do
      python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch $i
    done
elif [ ${dataset} = 'ctd' ]; then
    python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 3
elif [ ${dataset} = 'sider' ]; then
    python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 3
fi
