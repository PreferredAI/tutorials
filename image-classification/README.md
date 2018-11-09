The slides used for NUS High Workshop can be found at this [link](https://www.dropbox.com/s/im4lajfqifj3ozm/tutorial_v1_released.pdf?dl=0).

# Prerequisites

This code is written in Python 3 which can be downloaded from [here](https://www.python.org/downloads/release/python-355/). Please setup the environment if needed.

First, you need to clone our repository.

```bash
$ git clone https://github.com/PreferredAI/tutorials.git
$ cd tutorials/image-classification
```

Run command below to install required packages.

```bash
$ pip3 install -r requirements.txt
```

# Tutorial #1 - Facial Expression Recognition

## Dataset

[Facial Expression Recognition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) data consists of 48x48 pixel grayscale images of faces. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). \
In this tutorial, we adapt the original dataset to form a binary classification task which is categorizing each face in to one of two categories (0=Sad, 1=Happy).

We have provided a script to download the dataset. 

```bash
$ cd face-emotion
$ chmod +x download.sh | sh download.sh
```

If the script doesn't work, please use this [link](https://static.preferred.ai/tutorial/face-emotion/data.zip) to download the data. \
After the file is downloaded, you need to unzip it and move *data* folder into our tutorial folder *tutorials/image-classification/face-emotion*

The data is already split into training and testing sets with the statistics shown in table below.

| Class     | Training (# images) | Test (# images) |
| :-------: | :-----------------: | :-------------: |
| sad (0)   | 4347                | 483             |
| happy (1) | 4347                | 483             |
| **Total** | 8694                | 966             |

## Model

### Multilayer Perceptron (MLP)

The MLP architecture can be viewed as:

| Layer     | Dim/Kernel | Parameters                  |
| :-------: | :--------: | --------------------------: |
| fc1       | 128        | 48 x 48 x 128 + 1 = 294,913 |
| fc2       | 128        | 128 x 128 + 1 = 16,385      |
| output    | 2          | 128 x 2 + 1 = 257           |
| **Total** |            | 311,555                     |

### Shallow CNN

The shallow CNN architecture can be viewed as:

| Layer     | Dim/Kernel | Parameters                         |
| :-------: | :--------: | ---------------------------------: |
| conv      | 5 x 5      | 5 x 5 x 32 + 1 = 801               |
| pool      | 2 x 2      | 0                                  |
| fc        | 128        | 24 x 24 x 32 x 128 + 1 = 2,359,297 |
| output    | 2          | 128 x 2 + 1 = 257                  |
| **Total** |            | 2,360,355                          |


### Deep CNN

The deep CNN architecture can be viewed as:

| Layer     | Dim/Kernel | Parameters                      |
| :-------: | :--------: | ------------------------------: |
| conv1     | 5 x 5      | 5 x 5 x 32 + 1 = 801            |
| pool1     | 2 x 2      | 0                               |
| conv2     | 3 x 3      | 3 x 3 x 64 + 1 = 577            |
| pool2     | 2 x 2      | 0                               |
| conv3     | 3 x 3      | 3 x 3 x 128 + 1 = 1153          |
| conv4     | 3 x 3      | 3 x 3 x 128 + 1 = 1153          |
| pool3     | 2 x 2      | 0                               |
| fc        | 128        | 6 x 6 x 128 x 128 + 1 = 589,825 |
| output    | 2          | 128 x 2 + 1 = 257               |
| **Total** |            | 593,766                         |


## Training and Evaluation

Train model:

```bash
$ python3 src/train.py --model [model_name]
```

```
optional arguments:
  -h, --help                show this help message and exit
  --model                   MODEL
                              Type of CNN model (mlp or shallow or deep)
  --data_dir                DATA_DIR
                              Path to data folder (default: data)
  --log_dir                 LOG_DIR
                              Path to data folder (default: log)
  --checkpoint_dir          CHECKPOINT_DIR
                              Path to checkpoint folder (default: checkpoints)
  --num_checkpoints         NUM_CHECKPOINTS
                              Number of checkpoints to store (default: 1)
  --num_epochs              NUM_EPOCHS
                              Number of training epochs (default: 10)        
  --num_threads             NUM_THREADS
                              Number of threads for data processing (default: 2)
  --batch_size              BATCH_SIZE
                              Batch Size (default: 32)
  --dropout_rate            DROPOUT_RATE
                              Probability of dropping neurons (default: 0.5)
  --learning_rate           LEARNING_RATE
                              Learning rate (default: 0.001)
  --allow_soft_placement    ALLOW_SOFT_PLACEMENT
                              Allow device soft device placement
```


### Multilayer Perceptron

```bash
$ python3 src/train.py --model mlp
```

```text
Epoch number: 1
Training: 100%|███████████████████████████████████████████████████████████████████| 272/272 [00:10<00:00, 26.85it/s, loss=0.608]
train_loss = 0.6504, train_acc = 62.43 %
Testing: 100%|███████████████████████████████████████████████████████████████████| 31/31 [00:00<00:00, 37.88it/s, loss=0.536]
test_loss = 0.5852, test_acc = 70.29 %
Saved model checkpoint to checkpoints\mlp\epoch_1

...
...
...

Epoch number: 10
Training: 100%|███████████████████████████████████████████████████████████████████| 272/272 [00:09<00:00, 28.55it/s, loss=0.53]
train_loss = 0.5200, train_acc = 73.98 %
Testing: 100%|███████████████████████████████████████████████████████████████████| 31/31 [00:00<00:00, 35.52it/s, loss=0.18]
test_loss = 0.5059, test_acc = 73.40 %

Best accuracy = 75.67 %
```


### Shallow CNN

```bash
$ python3 src/train.py --model shallow
```

```text
Epoch number: 1
Training: 100%|███████████████████████████████████████████████████████████████████| 272/272 [00:33<00:00,  8.11it/s, loss=0.594]
train_loss = 0.6294, train_acc = 64.10 %
Testing: 100%|███████████████████████████████████████████████████████████████████| 31/31 [00:01<00:00, 22.09it/s, loss=0.396]
test_loss = 0.5600, test_acc = 71.12 %
Saved model checkpoint to checkpoints\shallow\epoch_1

...
...
...

Epoch number: 10
Training: 100%|███████████████████████████████████████████████████████████████████| 272/272 [00:33<00:00,  8.15it/s, loss=0.64]
train_loss = 0.3715, train_acc = 83.11 %
Testing: 100%|███████████████████████████████████████████████████████████████████| 31/31 [00:01<00:00, 20.50it/s, loss=0.108]
test_loss = 0.4408, test_acc = 78.88 %

Best accuracy = 79.50 %
```

### Deep CNN

```bash
$ python3 src/train.py --model deep
```

```text
Epoch number: 1
Training: 100%|███████████████████████████████████████████████████████████████████| 272/272 [01:00<00:00,  4.47it/s, loss=0.649]
train_loss = 0.6796, train_acc = 56.00 %
Testing: 100%|███████████████████████████████████████████████████████████████████| 31/31 [00:02<00:00, 13.47it/s, loss=0.639]
test_loss = 0.6664, test_acc = 58.39 %
Saved model checkpoint to checkpoints\deep\epoch_1

...
...
...

Epoch number: 10
Training: 100%|███████████████████████████████████████████████████████████████████| 272/272 [01:01<00:00,  4.41it/s, loss=0.374]
train_loss = 0.3226, train_acc = 85.82 %
Testing: 100%|███████████████████████████████████████████████████████████████████| 31/31 [00:02<00:00, 12.41it/s, loss=0.586]
test_loss = 0.2970, test_acc = 87.16 %
Saved model checkpoint to checkpoints\deep\epoch_10

Best accuracy = 87.16 %
```

## Test trained models with other images

Run test script with the model name and path to the folder containing test images:

```bash
$ python3 src/test.py --model [model_name] --data_dir [path_to_image_folder]
```

Some images are already in *test_images* folder for a quick check.

```bash
$ python3 src/test.py --model deep --data_dir test_images
```

We also host a [demo website](https://demo.preferred.ai/face-emotion) for you to try it out.


## Visualization

Here we visualize the Deep CNN model with saliency map of images using [Guided Backpropagation](https://arxiv.org/abs/1412.6806) technique, and the activation maps of the 4th convolutional layer of the network.

We may notice that the model focuses more on the mouth with Happy emotion, while other parts of the face are paid attention with Sad emotion.

| Input Image                                                                                                                      | Saliency Map                                                                                                                                  | Activation Maps of the 4th Convolutional Layer                                                                                       | Label |
| :------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------: | :---: |
| ![](https://raw.githubusercontent.com/PreferredAI/tutorials/master/image-classification/face-emotion/visualization/82.jpg)  | ![](https://raw.githubusercontent.com/PreferredAI/tutorials/master/image-classification/face-emotion/visualization/saliency_map_82.jpg)  | ![](https://raw.githubusercontent.com/PreferredAI/tutorials/master/image-classification/face-emotion/visualization/conv4_82.jpg)  | Happy |
| ![](https://raw.githubusercontent.com/PreferredAI/tutorials/master/image-classification/face-emotion/visualization/130.jpg) | ![](https://raw.githubusercontent.com/PreferredAI/tutorials/master/image-classification/face-emotion/visualization/saliency_map_130.jpg) | ![](https://raw.githubusercontent.com/PreferredAI/tutorials/master/image-classification/face-emotion/visualization/conv4_130.jpg) | Sad   |
| ![](https://raw.githubusercontent.com/PreferredAI/tutorials/master/image-classification/face-emotion/visualization/791.jpg) | ![](https://raw.githubusercontent.com/PreferredAI/tutorials/master/image-classification/face-emotion/visualization/saliency_map_791.jpg) | ![](https://raw.githubusercontent.com/PreferredAI/tutorials/master/image-classification/face-emotion/visualization/conv4_791.jpg) | Happy |
| ![](https://raw.githubusercontent.com/PreferredAI/tutorials/master/image-classification/face-emotion/visualization/607.jpg) | ![](https://raw.githubusercontent.com/PreferredAI/tutorials/master/image-classification/face-emotion/visualization/saliency_map_607.jpg) | ![](https://raw.githubusercontent.com/PreferredAI/tutorials/master/image-classification/face-emotion/visualization/conv4_607.jpg) | Sad   |

The code for visualization is also provided. You may need to install another additional package in order to generate the saliency maps.
```bash
$ pip3 install saliency
$ python3 src/visualize.py
``` 


# Tutorial #2 - Visual Sentiment Analysis

In this tutorial, the models are required a bit of time and computational resources to train. Thus, we provide trained models with a small set of testing data for quick evaluation. The code for training and testing are fully provided for reference purposes. More details can be found in the original [paper](https://www.researchgate.net/publication/320541140_Visual_Sentiment_Analysis_for_Review_Images_with_Item-Oriented_and_User-Oriented_CNN).

## Dataset and trained models

Run commands below to download the dataset and trained models.

```bash
$ cd vs-cnn
$ chmod +x download.sh | sh download.sh
```

## Base Model (VS-CNN)

Evaluate the base model on user dataset.

```bash
$ python3 src/eval_base.py --dataset user
```

```text
Boston
Loading data file: data/user/val_Boston.txt
Testing: 100%|███████████████████████████████████████████████████████████████████| 19/19 [00:36<00:00,  1.94s/it]
Pointwise Accuracy = 0.544
Pairwise Accuracy = 0.546

Avg. Pointwise Accuracy = 0.544
Avg. Pointwise Accuracy = 0.546
```


## Factor Model for User (uVS-CNN)

Evaluate the factor model on user dataset.

```bash
$ python3 src/eval_factor.py --dataset user
```

```text
Boston
Loading data file: data/user/val_Boston.txt
Testing: 100%|███████████████████████████████████████████████████████████████████| 1194/1194 [00:48<00:00, 24.41it/s]
Pointwise Accuracy = 0.664
Pairwise Accuracy = 0.720

Avg. Pointwise Accuracy = 0.664
Avg. Pointwise Accuracy = 0.720
```
