# RNN

## Link
* Task : https://docs.google.com/presentation/d/1W5-D0hqchrkVgQxwNLBDlydamCHx5yetzmwbUiksBAA/edit#slide=id.g7cd4f194f5_2_226
* Result : [**report.pdf**](report.pdf)

## Usage 
* train model
```zsh
bash train.sh train_data train_unlabel_data
// ex : bash train.sh data/training_label.txt  data/training_nolabel.txt
// data is available at https://www.kaggle.com/c/ml2020spring-hw4
```

* test model 
```zsh
bash test.sh test_data out_file
// ex: bash test.sh data/testing_data.txt out.csv
```
