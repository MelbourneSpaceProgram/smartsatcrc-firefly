# SmartSat CRC Ideation Challange 01: Firefly
This is a repository containing code for training and implementing machine learning algorithms for the SmartSat Ideation Challenge 01: Firefly. The algorithms are implemented in Python using the Pytorch framework and trained using Amazon Sagemaker. The algorithms comprise of various segmenation models for the purpose of identifying and geographically isolating bushfire smoke. This is then used to estimate the effects of signal attenuation through smoke and fire -- a known problem with the ability to create "signal backspots".
The models are to be deployed on the NVIDIA Jetson Nano to form a payload on a stratospheric balloon, with the aim of providing real-time analysis of hyperspectral imagery.

# The Project MVP
A Machine Learning model that predicts communication blackout areas caused by bush-fire and smoke using active cell tower location and multispectral imagery.

# Repository Outline
## Classification
As an initial step, the team developed classification models that can differentiate between smoke and non-smoke images. Model development was motivated by the study entitled "SmokeNet: Satellite Smoke Scene Detection Using Convolutional Neural Network with Spatial and Channel-Wise Attention". Using the data made public by the paper's authors ([link to authors' page with dataset](http://complex.ustc.edu.cn/2019/0802/c18202a389656/page.htm), the team deployed various convolutional neural networks --  afforded by the Pytorch library. As a result, a greater familiarity with the framework was developed and the potential to utilise classification as a pre-processing step in the payload. 

## Segmentation
### Model Testing
The first step of developing the segmentation model was to find existing Pytorch implementions of effective architectures. Although UNets were considered, the team focused on Segnets because the former is used primarily for medical imaging. Provided more time, a wider selection of architecures will be tested.

The models were adopted from publically available and MIT liscenced repositories and asigned the following numbers:
* SegNet1 (([repo link](https://github.com/trypag/pytorch-unet-segnet))
* SegNet2 (([repo link](https://github.com/delta-onera/segnet_pytorch))
* SegNet6 (([repo link](https://github.com/say4n/pytorch-segnet))

Evaluation was conducted on cloud imagery adopted from a project that inquired about the effectiveness of UNets for cloud segmentation (([link](https://www.kaggle.com/cordmaur/38-cloud-simple-unet)). The data comprised of imagery with 4 input channels -- red, green, blue and near infrared -- and target masks that identified the presence of cloud in the image. It was assumed that testing the SegNets on this data would reliably reflect the models' generalisability to bushfire smoke. 

### Model Training 
Model training was carried out on a combination of using Google Colab, Google Cloud Platform instances and Amazom Sagemaker. The first two were used in the begninning as the team had prior experience with these platforms. As the project progressed, Amazon Sagemaker was adopted as the primary platform for training, hyperparameter tuning and hosting data via S3 buckets. 

### Smoke Segmentation Data
Imagery captured on the Sentinal 2 was used to collect a dataset of bushfire imagery. This 12 channel mutlispectural imagery was downloaded and labelled via LabelBox. Given that the images are thouasands of pixels tall and wide, they are to be sampled from using a croping size of approximately 300 by 300 pixels, depending on the model being used. Prior to combining the spectral channels, they were required to be interpolated to have the same resolution. The models then are to be trained using random cropping and various data augmentation techniques such as rotation and random flipping. 




