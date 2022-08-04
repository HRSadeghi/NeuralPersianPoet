## Neural Persian Poet: A sequence-to-sequence model for composing Persian poetry



### Installation
We provide instructions how to install dependencies via  pip.
First, clone the repository locally:

```
git https://github.com/HRSadeghi/NeuralPersianPoet.git
```

Change the current directory to NeuralPersianPoet:
```
cd NeuralPersianPoet
```

Now, prepare environment:
```
pip install -r requirements.txt
```



### Data Preparation

In this project, the ganjoor dataset is used, which includes ancient and contemporary Persian poems. All these poems are available online at [ganjoor website](https://ganjoor.net/). All information related to poems, poets, type of the poems etc. has been collected in a SQL database. In order to be able to use this data, we have collected the important information of this database in the form of a .csv dataset. You can download this dataset from [here](https://drive.google.com/drive/folders/1ZdB8A6i_y5LUaAFSy96GZdkM9o8Stipm?usp=sharing).

After adding the dataset directory, we expect the structure of the project to be as follows:

```
NeuralPersianPoet
├── dataset
│   ├── ganjoor.csv
│   ├── poet_list_train.pickle
│   ├── poet_list_test.pickle
│   ├── poem_list_train.pickle
│   ├── poem_list_test.pickle
├── models
...
```

If you want to use the customized dataset, you can first convert your poems' file to the ganjoor.csv file format and place it in the dataset directory, and then run the following code snippet.

```python
from dataLoader.utils import load_file, save_file
from dataLoader.utils import get_poems_and_poets
from sklearn.model_selection import train_test_split

poem_list, poet_list  = get_poems_and_poets(path = "PATH/To/YOUR_CUSTOM_DATASET",
                                            cleaning = True,
                                            constrained_poets = ['poet1','poet2', ...]  # like ['حافظ','فردوسی']
                                            )

poem_list_train, poem_list_test, poet_list_train, poet_list_test = train_test_split(poem_list, poet_list, test_size=0.3, shuffle=False)

save_file(poem_list_train, 'dataset/poem_list_train.pickle')
save_file(poem_list_test, 'dataset/poem_list_test.pickle')
save_file(poet_list_train, 'dataset/poet_list_train.pickle')
save_file(poet_list_test, 'dataset/poet_list_test.pickle')
```

### Tokenizer Preparation
Like the GPT model, Byte Pair Encoding (BPE) Tokenizer is used here to tokenize the poems. To use this tokenization, we need to train it. 




### Training

Training of the model 

### Inference

```
python inference.py 
```

