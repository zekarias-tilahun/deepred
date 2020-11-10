dataset=${1}
root_dir=./data/${dataset}/

if [ ${dataset} = 'lastfm' ] || [ ${dataset} = 'reddit' ] || [ ${dataset} = 'wikipedia' ]; then
    python deepred/preprocess.py --root-dir ${root_dir} --temporal --tr-rate .8 --dev-rate .1
else
    python deepred/preprocess.py --root-dir ${root_dir} --static --tr-rate .6 --dev-rate .1 --sample
fi
