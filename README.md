# Preserving Fairness Generalization in Deepfake Detection

Li Lin, Xinan He, Yan Ju, Xin Wang, Feng Ding, and Shu Hu
_________________

This repository is the official implementation of our paper "Preserving Fairness Generalization in Deepfake Detection", which has been accepted by **CVPR 2024**. 

## 1. Installation
You can run the following script to configure the necessary environment:

```
cd Fairness-Generalization
conda create -n FairnessDeepfake python=3.9.0
conda activate FairnessDeepfake
pip install -r requirements.txt
```

## 2. Dataset Preparation

- Download [FF++](https://github.com/ondyari/FaceForensics), [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics), [DFD](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html) and [DFDC](https://ai.facebook.com/datasets/dfdc/) datasets
- Download annotations for these four datasets according to [paper](https://arxiv.org/pdf/2208.05845.pdf) and their [code](https://github.com/pterhoer/DeepFakeAnnotations), extract the demographics information of all images in each dataset. 
- Extract, align and crop face using [DLib](https://www.jmlr.org/papers/volume10/king09a/king09a.pdf), and save them to `/path/to/cropped_images/`
- Split cropped images in each dataset to train/val/test with a ratio of 60%/20%/20% without identity overlap.
- Generate faketrain.csv, realtrain.csv, fakeval.csv, realval.csv according to following format:
  
		|- faketrain.csv
			|_ img_path,label,ismale,isasian,iswhite,isblack,intersec_label,spe_label
				/path/to/cropped_images/imgxx.png, 1(fake), 1(male)/-1(not male), 1(asian)/-1(not asian), 1(black)/-1(not black), 1(white)/-1(not white), 0(male-asian)/1(male-white)/2(male-black)/3(male-others)/4(female-asian)/5(female-white)/6(female-black)/7(female-others), 1(Deepfakes)/2(Face2Face)/3(FaceSwap)/4(NeuralTextures)/5(FaceShifter)
				...

		|- realtrain.csv
			|_ img_path,label,ismale,isasian,iswhite,isblack,intersec_label
				/path/to/cropped_images/imgxx.png, 0(real), 1(male)/-1(not male), 1(asian)/-1(not asian), 1(black)/-1(not black), 1(white)/-1(not white), 0(male-asian)/1(male-white)/2(male-black)/3(male-others)/4(female-asian)/5(female-white)/6(female-black)/7(female-others)
				...

		|- fakeval.csv
			|_ img_path,label,ismale,isasian,iswhite,isblack,intersec_label,spe_label
				/path/to/cropped_images/imgxx.png, 1(fake), 1(male)/-1(not male), 1(asian)/-1(not asian), 1(black)/-1(not black), 1(white)/-1(not white), 0(male-asian)/1(male-white)/2(male-black)/3(male-others)/4(female-asian)/5(female-white)/6(female-black)/7(female-others), 1(Deepfakes)/2(Face2Face)/3(FaceSwap)/4(NeuralTextures)/5(FaceShifter)
				...

		|- realval.csv
			|_ img_path,label,ismale,isasian,iswhite,isblack,intersec_label
				/path/to/cropped_images/imgxx.png, 0(real), 1(male)/-1(not male), 1(asian)/-1(not asian), 1(black)/-1(not black), 1(white)/-1(not white), 0(male-asian)/1(male-white)/2(male-black)/3(male-others)/4(female-asian)/5(female-white)/6(female-black)/7(female-others)
				...
		
- Generate test.csv according to following format:

		|- test.csv
			|- img_path,label,ismale,isasian,iswhite,isblack,intersec_label
				/path/to/cropped_images/imgxx.png, 1(fake)/0(real), 1(male)/-1(not male), 1(asian)/-1(not asian), 1(black)/-1(not black), 1(white)/-1(not white), 0(male-asian)/1(male-white)/2(male-black)/3(male-others)/4(female-asian)/5(female-white)/6(female-black)/7(female-others)
				...

## 3. Load Pretrained Weights
Before running the training code, make sure you load the pre-trained weights. We provide pre-trained weights under [`./training/pretrained`](./training/pretrained). You can also download *Xception* model trained on ImageNet (through this [link](http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth)) or use your own pretrained *Xception*.

## 4. Train
To run the training code, you should first go to the [`./training/`](./training/) folder, then you can train our detector with loss flattening strategy by running [`train.py`](training/train.py), or without loss flattening strategy by running [`train_noSAM.py`](training/train_noSAM.py):

```
cd training

python train.py 
```

You can adjust the parameters in [`train.py`](training/train.py) to specify the parameters, *e.g.,* training dataset, batchsize, learnig rate, *etc*.

`--lr`: learnig rate, default is 0.0005. 

`--gpu`: gpu ids for training.

` --fake_datapath`: /path/to/faketrain.csv, fakeval.csv

` --real_datapath`: /path/to/realtrain.csv, realval.csv

`--batchsize`: batch size, default is 16.

`--dataname`: training dataset name: ff++.

`--model`: detector name: fair_df_detector.

## 5. Test
* For model testing, we provide a python file to test our model by running `python test.py`. 

	`--test_path`: /path/to/test.csv 

	`--test_data_name`: testing dataset name: ff++, celebdf, dfd, dfdc.

	`--inter_attribute`: intersectional group names divided by '-': male,asian-male,white-male,black-male,others-nonmale,asian-nonmale,white-nonmale,black-nonmale,others 

	`--single_attribute`: single attribute name divided by '-': male-nonmale-asian-white-black-others 

	`--checkpoints`: /path/to/saved/model.pth 

	`--savepath`: /where/to/save/predictions.npy(labels.npy)/results/ 

	`--model_structure`: detector name: fair_df_detector.

	`--batch_size`: testing batch size: default is 32.

* After testing, for metric calculation, we provide `python fairness_metrics.py` to print all the metrics. To be noted that before run metrics.py, adjust the input to the path of your predictions(labels).npy files, which is the `--savepath` in the above setting.

#### 📝 Note
Change `--inter_attribute` and `--single_attribute` for different testing dataset:

```
### ff++, dfdc
--inter_attribute male,asian-male,white-male,black-male,others-nonmale,asian-nonmale,white-nonmale,black-nonmale,others \
--single_attribute male-nonmale-asian-white-black-others \

### celebdf, dfd
--inter_attribute male,white-male,black-male,others-nonmale,white-nonmale,black-nonmale,others \
--single_attribute male-nonmale-white-black-others \
```

## 📦 Provided Backbones
|                  | File name                               | Paper                                                                                                                                                                                                                                                                                                                                                         |
|------------------|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Xception          | [xception.py](./training/networks/xception.py)         | [Xception: Deep learning with depthwise separable convolutions](https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html) |
| ResNet50          | [resnet50.py](training/networks/resnet50.py)       | [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)                                                                                                                                                                                                                                                                                              |
| EfficientNet-B3      | [efficientnetb3.py](./training/networks/efficientnetb3.py) | [Efficientnet: Rethinking model scaling for convolutional neural networks](http://proceedings.mlr.press/v97/tan19a.html)                                                                                                                                                                                                                  |
| EfficientNet-B4      | [efficientnetb4.py](./training/networks/efficientnetb4.py) | [Efficientnet: Rethinking model scaling for convolutional neural networks](http://proceedings.mlr.press/v97/tan19a.html) 

## Citation
Please kindly consider citing our papers in your publications. 
```bash
@inproceedings{Li2024preserving,
    title={Preserving Fairness Generalization in Deepfake Detection},
    author={Li Lin, Xinan He, Yan Ju, Xin Wang, Feng Ding, Shu Hu},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024},
}
```
