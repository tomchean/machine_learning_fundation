# Adversarial Attack

## Link
* Task : https://docs.google.com/presentation/d/1dQVeHfIfUUWxMSg58frKBBeg2OD4N7SD0YP3LYMM7AA/edit#slide=id.g7be340f71d_0_0
* Result : [**report.pdf**](report.pdf)

## Usage 
* use logistct model
```zsh
bash logistic.sh train_data testing_data train_feature train_label test_feature out_path
// ex: bash logistic.sh data/train.csv data/test.csv data/X_train data/Y_train data/X_test output.csv
```

* use generative model
```zsh
bash negerative.sh train_data testing_data train_feature train_label test_feature out_path
// ex: bash generative.sh data/train.csv data/test.csv data/X_train data/Y_train data/X_test output.csv
```
