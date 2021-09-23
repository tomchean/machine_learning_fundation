# GAN

## Link
* Task : https://docs.google.com/presentation/d/1_6TJrFs3JGBsJpdRGLK1Fy_EiJlNvLm_lTZ9sjLsaKE/edit#slide=id.p4
* Result : [**report.pdf**](report.pdf)

## Usage 
* test model
```zsh
bash test_wgan/test_dcgan.sh checkpoint output
//ex : bash test_dcgan.sh checkpoints/dcgan.pth out_img
```

* train model
```zsh
bash train.sh data_dir checkpoint
// ex : bash train.sh data checkpoint/dcgan.pth
// data is available at https://drive.google.com/uc?id=1IGrTr308mGAaCKotpkkm8wTKlWs9Jq-p
```
