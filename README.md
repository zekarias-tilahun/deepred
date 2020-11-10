# DeepRed
A PyTorch implementation of DeepRed, an algorithm for next item prediction in temporal interaction networks.


Requirements!
-------------
  - Python 3.6+
  - PyTorch 1.4+
  - Numpy 1.17.2+
  - Networkx 2.3+

Example usage
-------------

To run the algorithm, a dataset should be placed under the ```./data/<DATASET>/``` directory, where 
```DATASET``` is the name of a dataset and it could be, ```reddit```, ```wikipedia```, ```lastfm```, ```matador```, ```sider```, ```steam```.
Any dataset other than these can be used, so long as the following formats are respected.

Please ensure that the ```./data/<DATASET>/``` directory contains

```
./data/<DATASET>/processed_dir/train.txt
./data/<DATASET>/processed_dir/dev.txt
./data/<DATASET>/processed_dir/test.txt
./data/<DATASET>/model_dir/
./data/<DATASET>/output_dir/
```

The [```preprocessing```](#Preprocessing) step can be used to generate the above directories and files.
The format of the train, dev, or test files should be one triple per line, as
```
user_id item_id timestamp
```
containing the ids of a user and an item interacting at a certain timestamp.


Once we setup the above files and directories, then deepred can be executed using 

```sh
$ python ./deepred/deepred.py
```

or 

```sh
$ bash run.sh <DATASET>
```

```./deepred/deepred.py``` takes the following input arguments

`--root-dir:`
A path to the directory assosiated to the ```DATASET``` under consideration. Default is ```./data/wikipedia```

`--nbr-size:`
The neighborhood size or the value for k in the paper. Default is 100.

`--dim:`
The size of the embedding dimension. Default is 128.

`--lr:`
Learning rate. Default is 0.0001

`--reg-cof:`
A regularization coefficient to avoid collapse into a subspace or $\gamma$ in the paper. Default is 0.7

`--dropout:`
A dropout rate to avoid overfitting. Default is 0.5

`--epochs:`
The number of epochs. Default is 2.

`--temporal:`
A flag to indicate that ```DATASET``` is a temporal interacton network. Default is True.

`--static:`
A flag to indicate that ```DATASET``` is a static interacton network. Default is False.


`--log-level:`
The log level. The smaller the log level the more verbose it is. Values are between 0 and 4. Default is 0.


Preprocessing
-------------

The aforementioned files used for running deepred are generated from a single file ```./data/<DATASET>/raw_dir/interaction.txt```
To this end, the deepred/preprocess.py module can be used. The ```./data/<DATASET>/raw_dir/interaction.txt``` file should contain 
triples of the form mentioned above.

The preprocessing takes the same arguments `--root-dir:`, `--temporal:`, `--static:`, `--log-level:` mentioned above in addition
to the following ones.

`--tr-rate:`
Training rate (proportion). Default is 0.8

`--dev-rate:`
Development rate (proportion). Default is 0.1


Evaluation
----------

To evaluate the performance of the algorithm use 



```sh
$ python ./deepred/next_item_prediction.py  --root-dir ./data/<DATASET>/ --epoch <EPOCH>
```

where ```--epcoh``` specifies the epoch to be evaluated.

or 

```sh
$ bash eval.sh <DATASET>
```

Application to static interaction networks.
---

To apply DeepRed to static interaction networks, simply change the format of the input data as

```user_id item_id```

and activate the ```--static``` flag whenever necessary.

Evaluate using

```sh
python deepred/interaction_prediction.py --root-dir ./data/<DATASET>/ --epoch 6
```