# Prerequisites

This code is written in [Python3](https://www.python.org/downloads/) and requires [TensorFlow 1.7+](https://www.tensorflow.org/install/). In addition, you need to install a few more packages to process data, [Anaconda](https://www.anaconda.com/download/) is recommended for python environment.

First, you need to clone our repository.

```bash
$ git clone https://github.com/PreferredAI/Tutorials.git
$ cd Tutorials/1806-image-classification
```

Run command below to install some required packages.

```bash
$ pip3 install -r requirements.txt
```

# Tutorial #1 - Facial Expression Recognition

## Dataset

In this tutorial, we provide data consisting of 48x48 pixel grayscale images of faces. The task is to categorize each face based on the emotion shown in the facial expression in to one of two categories (0=Sad, 1=Happy).

I have provided a script to download the dataset. 

```bash
$ cd face-emotion
$ chmod +x download.sh | sh download.sh
```

The data is already split into training and testing sets with the statistics shown in table below.

| Class       | Training (# images) | Test (# images) |
| :---------: | :-----------------: | :-------------: |
| happy (1)   | 4347                | 483             |
| sad (0)     | 4347                | 483             |
| **Total**   | 8694                | 966             |

## Model

### Multilayer Perceptron (MLP)

The MLP architecture can be viewed as:

| Layer     | Dim/Kernel | Parameters                  |
| :-------: | :--------: | --------------------------: |
| fc1       | 512        | 48 x 48 x 512+1 = 1,179,649 |
| fc2       | 512        | 512 x 512+1 = 262,145       |
| output    | 2          | 512 x 2+1 = 1025            |
| **Total** |            | 1,442,819                   |

### Shallow CNN

The shallow CNN architecture can be viewed as:

| Layer     | Dim/Kernel | Parameters                       |
| :-------: | :--------: | -------------------------------: |
| conv      | 5 x 5      | 5 x 5 x 32+1 = 801               |
| pool      |            | 0                                |
| fc        | 512        | 24 x 24 x 32 x 512+1 = 9,437,185 |
| output    | 2          | 512 x 2+1 = 1025                 |
| **Total** |            | 9,439,011                        |


### Deep CNN

The deep CNN architecture can be viewed as:

| Layer     | Dim/Kernel | Parameters                        |
| :-------: | :--------: | --------------------------------: |
| conv1     | 5 x 5      | 5 x 5 x 32 + 1 = 801              |
| pool1     |            | 0                                 |
| conv2     | 3 x 3      | 3 x 3 x 64 + 1 = 577              |
| pool2     |            | 0                                 |
| conv3     | 3 x 3      | 3 x 3 x 128 + 1 = 1153            |
| conv4     | 3 x 3      | 3 x 3 x 128 + 1 = 1153            |
| pool3     |            | 0                                 |
| fc        | 512        | 6 x 6 x 128 x 512 + 1 = 2,359,397 |
| output    | 2          | 512 x 2 + 1 = 1025                |
| **Total** |            | 2,364,006                         |


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
train_loss = 0.6452, train_acc = 63.61 %
Testing: 100%|███████████████████████████████████████████████████████████████████| 31/31 [00:00<00:00, 37.88it/s, loss=0.536]
test_loss = 0.5819, test_acc = 68.74 %

...
...
...

Epoch number: 10
Training: 100%|███████████████████████████████████████████████████████████████████| 272/272 [00:09<00:00, 28.55it/s, loss=0.53]
train_loss = 0.4391, train_acc = 79.03 %
Testing: 100%|███████████████████████████████████████████████████████████████████| 31/31 [00:00<00:00, 35.52it/s, loss=0.18]
test_loss = 0.5077, test_acc = 74.53 %
Saved model checkpoint to checkpoints/mlp/epoch_10

Best accuracy = 74.53 %
```


### Shallow CNN

```bash
$ python3 src/train.py --model shallow
```

```text
Epoch number: 1
Training: 100%|███████████████████████████████████████████████████████████████████| 272/272 [00:33<00:00,  8.11it/s, loss=0.594]
train_loss = 0.6214, train_acc = 65.38 %
Testing: 100%|███████████████████████████████████████████████████████████████████| 31/31 [00:01<00:00, 22.09it/s, loss=0.396]
test_loss = 0.5418, test_acc = 73.19 %

...
...
...

Epoch number: 10
Training: 100%|███████████████████████████████████████████████████████████████████| 272/272 [00:33<00:00,  8.15it/s, loss=0.64]
train_loss = 0.2343, train_acc = 90.40 %
Testing: 100%|███████████████████████████████████████████████████████████████████| 31/31 [00:01<00:00, 20.50it/s, loss=0.108]
test_loss = 0.4628, test_acc = 78.67 %

Best accuracy = 78.88 %
```

### Deep CNN

```bash
$ python3 src/train.py --model deep
```

```text
Epoch number: 1
Training: 100%|███████████████████████████████████████████████████████████████████| 272/272 [01:00<00:00,  4.47it/s, loss=0.649]
train_loss = 0.6766, train_acc = 56.68 %
Testing: 100%|███████████████████████████████████████████████████████████████████| 31/31 [00:02<00:00, 13.47it/s, loss=0.639]
test_loss = 0.6473, test_acc = 62.94 %

...
...
...

Epoch number: 10
Training: 100%|███████████████████████████████████████████████████████████████████| 272/272 [01:01<00:00,  4.41it/s, loss=0.374]
train_loss = 0.3042, train_acc = 86.71 %
Testing: 100%|███████████████████████████████████████████████████████████████████| 31/31 [00:02<00:00, 12.41it/s, loss=0.586]
test_loss = 0.3569, test_acc = 84.06 %

Best accuracy = 86.13 %
```

## Test trained models with other images

Run test script with the model name and path to the folder containing test images:

```bash
$ python3 src/test.py --model [model_name] --data_dir [path_to_image_folder]
```

Some images are already in *test_images* folder for a quick run.

```bash
$ python3 src/test.py --model deep --data_dir test_images
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

Evaluate trained base model on user dataset.

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

Evaluate trained factor model on user dataset.

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