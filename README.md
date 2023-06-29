# plant_disease_classification
In this repo, you can find the dataset from Kaggle for plant disease, download and do training 
https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset/code?resource=download

then you can run this command to d otraining. 

python train.py --batch_size 32 --num_epochs 10 --hidden_units 10 --lr 0.001 --device cuda --train_dir .\DATASET_KAGGLE\archive\Train\Train\ --test_dir .\DATASET_KAGGLE\archive\Test\Test\
