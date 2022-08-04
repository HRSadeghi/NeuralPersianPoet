## Neural Persian Poet: A sequence-to-sequence model for composing Persian poetry



### Installation
We provide instructions how to install dependencies via conda.
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

In this project, the ganjoor dataset is used, which includes ancient and contemporary Persian poems. All these poems are available online at [ganjoor website](https://ganjoor.net/). All information related to poems, poets, etc. has been collected in a SQL database. In order to be able to use this data, we have collected the important information of this database in the form of a .csv file. You can download this dataset from [here](https://drive.google.com/drive/folders/1ZdB8A6i_y5LUaAFSy96GZdkM9o8Stipm?usp=sharing).

We expect the directory structure to be the following:
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






### Training

Training of the model 

### Inference

```
python inference.py 
```

