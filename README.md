# img-classification-master-using-pytorch
A simple demo of image classification using pytorch. Here, we use a custom dataset containing 43956 images belonging to 11 classes for training(and validation). Also, we compare three different approaches for training viz. training from scratch, finetuning the convnet and convnet as a feature extractor, with the help of pretrained pytorch models. The models used include: VGG11, Resnet18 and MobilenetV2.

Dependencies
Python3, Scikit-learn
Pytorch, PIL
Torchsummary, Tensorboard
pip install torchsummary # keras-summary
pip install tensorboard  # tensoflow-logging
NB: Update the libraries to their latest versions before training.

Run the following scripts for training and/or testing

python train.py # For training the model [--mode=finetune/transfer/scratch]
python test.py test # For testing the model on sample images
python eval.py eval_ds # For evaluating the model on new dataset
Training results
Accuracy	Size	Training Time	Training Mode
VGG11	96.73	515.3 MB	900 mins	scratch
Resnet18	99.85	44.8 MB	42 mins	finetune
MobilenetV2	97.72	9.2 MB	32 mins	transfer
Batch size: 64, GPU: Tesla K80

Both Resnet18 and MobilenetV2(transfer leraning) were trained for 10 epochs; whereas VGG11(training from scratch) was trained for 100 epochs.

Training graphs
Resnet18:-

Finetuning the pretrained resnet18 model. Screenshot

Mobilenetv2:-


Mobilenetv2 as a fixed feature extractor. Screenshot

Sample outputs
Sample classification results

Screenshot

Evaluation
Here we evaluate the performance of our best model - resnet18 on a new data-set containing 50 images per class.
Accuracy of the network on the 550 test images: 99.09%
Per class accuracy

Accuracy of class    apple : 100.00 %
Accuracy of class atm card : 100.00 %
Accuracy of class   camera : 100.00 %
Accuracy of class      cat : 100.00 %
Accuracy of class   banana : 100.00 %
Accuracy of class   bangle : 92.00 %
Accuracy of class  battery : 98.00 %
Accuracy of class   bottle : 100.00 %
Accuracy of class    broom : 100.00 %
Accuracy of class     bulb : 100.00 %
Accuracy of class calender : 100.00 %
