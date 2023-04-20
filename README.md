repo structure:

This is the structure of the relevant items you may want to look at in the repo. I have organized them here. The only other documents are image files generated from running the smaller scripts. 


#### Final_master_notebook

#### Papyrus_presentation

#### README

#### gitignore

#### submission

----notebooks

--------images_output

--------vesuvius_notebook

--------classifer_ai_letters

------------letter_class

----------------data_classifier

--------------------train_high_resolution

--------------------test_high_resolution

--------------------test

--------------------train


# Project Description:


## Data source for Vesuvius Challenge: https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/overview

- This Data Source is suitable for the project because it defines the project. This data is directly from the competition hosts for us to use and try to work with in order to increase the legibility of the papyrus scrolls. 

## size = 37.02 GB

files = 340

### From the competition website

Files
[train/test]/[fragment_id]/surface_volume/[image_id].tif slices from the 3d x-ray surface volume. Each file contains a greyscale slice in the z-direction. Each fragment contains 65 slices. Combined this image stack gives us width * height * 65 number of voxels per fragment. You can expect two fragments in the hidden test set, which together are roughly the same size as a single training fragment. The sample slices available to download in the test folders are simply copied from training fragment one, but when you submit your notebook they will be substituted with the real test data.

[train/test]/[fragment_id]/mask.png — a binary mask of which pixels contain data.

train/[fragment_id]/inklabels.png — a binary mask of the ink vs no-ink labels.

train/[fragment_id]/inklabels_rle.csv — a run-length-encoded version of the labels, generated using this script. This is the same format as you should make your submission in.

train/[fragment_id]/ir.png — the infrared photo on which the binary mask is based.

sample_submission.csv, an example of a submission file in the correct format. You need to output the following file in the home directory: submission.csv. See the evaluation page for information.


## Data source for Handwritten Greek Letters: https://www.kaggle.com/datasets/katianakontolati/classification-of-handwritten-greek-letters

- This Data Source is suitable for the project because it is used to make a Handwritten Greek Letter Classifier 2D Convolutional Neural Network which can in theory take inputs from the outputs of the Ink Detection Neural Network and classify those outputs as a particular Greek letter. This aids me in cleaning the final image, as I can use this second network to help increase the pixel values that are the most shared with the letter it is trying to predict. 

## size = 23 MB

files =

    train = 240
    
    test = 96

### From the website

Content
The training dataset consists of 240 images of Greek letters (10 for each letter). The test dataset consists of 96 images (4 for each letter). We provide both the grayscale original high-resolution images in 2 zip files (train and test) and the low-resolution images (14x14 pixels) used for the classification in another 2 zip files (train and test). Finally, matrices that correspond to each low-resolution (14x14) image are given in 2 .csv files (train and test). Numbers in cells represent the grayscale intensities.

The last column in both .csv files contains the labels/ ground truth, that is the 24 different letters. Each letter corresponds to a number in a way that it is explained below:

Letter Symbols => Letter Labels
α=>1, β=>2, γ=>3, δ=>4, ε=>5, ζ=>6, η=>7, θ=>8, ι=>9, κ=>10,
λ=>11, μ=>12, ν=>13, ξ=>14, ο=>15, π=>16, ρ=>17, σ=>18, τ=>19, υ=>20,
φ=>21, χ=>22, ψ=>23, ω=>24

In summary we have:

1. The original high-resolution images
(train _high _resolution.zip, test _high _resolution.zip)

2. The low-resolution (14x14) images
(train _letters _images.zip, test _letters _images.zip)

3. Training dataset
Grayscale intensities- with 240 rows/data, 196 columns/features, column 197 contains the labels (train.csv)

4. Test dataset
Grayscale intensities- with 96 rows/data, 196 columns/features, column 197 contains the labels (test.csv)

Tip: Only the .csv files are needed for the classification, the images are for illustration purposes.

## Inclusion of features justification

#### Vesuvius Challenge Dataset

I include these 10 slices in this position in the z axis of the papyrus scrolls because according to others online who are also competing, these are the only relevant slices with enough ink worth going through. Other slices are excluded according to this code:

BUFFER = 30  # Buffer size in x and y direction
Z_START = 27 # First slice in the z direction to use
Z_DIM = 10   # Number of slices in the z direction

Also, I did a lot of image processing I discuss further in the notebook and used Kuwahara versions of the raw data to train the model, decreasing training time significantly. 

#### Handwritten Greek Letters Dataset

I reformatted the original .csv table the data comes with to make it useable. I opted to select the high resolution images for my training data and scaled them dowm to 66 * 66 using datagen. I found a hard time teaching the network with the base model using the base 14 * 14 images. Greek letters are extremely hard to classify becuase many of them look similar or have complex forms, which is why I only used the higher resolution images for the final model. 

## Data Limitations 

The Vesuvius Challenge dataset is considerably large, with a size of 37.02 GB, which leads to issues with data storage and processing. The availability of only two fragments in the hidden test set limits the robustness of the developed model. 

The Handwritten Greek Letters dataset has a very small number of training and test images, which results in a failure to converge due to some overfitting and poor generalization performance. Finally, the use of high resultion images increased training and rendering time making it hard to perfect both Neural Networks at once. 


# Modeling

#### Vesuvius Challenge Dataset

In this project I iterated improving on the base Vesuvius Challenge Model by making the network more normalized with Dropouts, Batch Normalization, MaxPooling, Dense Layers, various optimizers, and different learning rates. I added batch normalization and dropouts first, which did not work well together, followed by adding dense layers. I found that image pre-processing combined with batch normalization and only dropouts on certain layers helped the most. Adding Dense layers did not change anything. In the end our model only kept some changes, but is definetly an improvement upon the base model. I am able to generate a comparable image with 1/3 of the training time. I tried optimizers such as Adadelta, Adamax, and RMSprop but the SGD optimizer performed the best for this task.

Baseline Vesuvius Model:

model = nn.Sequential(
    nn.Conv3d(1, 16, 3, 1, 1), nn.MaxPool3d(2, 2),
    nn.Conv3d(16, 32, 3, 1, 1), nn.MaxPool3d(2, 2),
    nn.Conv3d(32, 64, 3, 1, 1), nn.MaxPool3d(2, 2),
    nn.Flatten(start_dim=1),
    nn.LazyLinear(128), nn.ReLU(),
    nn.LazyLinear(1), nn.Sigmoid()
).to(DEVICE)

The first section of this notebook is adapted from the starting code needed to render the data, package it for use in the Ink Detection Neural Network, and package the final output file for submission to the competition. This open source tutorial was a great starting point for this project I am citing the source here: https://www.kaggle.com/code/jpposma/vesuvius-challenge-ink-detection-tutorial

Of course I made modifications wherever I could make improvements, but this saved me a lot of data preprocessing and postprocessing work and let me focus on optimizing the network and output instead via image processing and the Letter Classifer AI. 

#### Handwritten Greek Letters Dataset

The Handwritten Greek Letters dataset, also obtained from Kaggle is used to make a secondary Handwritten Greek Letter Classifier 2D Convolutional Neural Network. I use the high-resolution images of the training data and scale them for the model.

I started with a very simple neural network as a base model, the same one commonly used on the MNIST dataset. Something like this:

model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

        https://www.kaggle.com/code/prashant111/mnist-deep-neural-network-with-keras


One of the first things I did was adding in 2D convolutions, which significantly helped. Once I optimized the depth and density of the network, I optimized the convoutional filters to their optimal state. The final model I created appears to score very low, but in fact was useful enough to prove my concept might work if given more time and resources. The reason it scores so low is the datagen parameters I set are very variable to the images, and distort them a lot. First I discovered my model was tuned with much less rigorous datagen parameters when it was scoring 80% roughly. Then i stopped moving the filters and optimizers, and instead make the dataset much harder for it. This worked out well in the end, as it works well enough to show that my overlay technique may be useful. I tried optimizers such as Adadelta, Adamax, and RMSprop, but Adam worked the best. 

## Conclusion

This notebook serves as a proof of concept for some ideas worth exploring further for the Vesuvius Challenge. I demonstrate that image enhancement techniques can be applied to improve the outputs from the Ink Detection AI, improving upon the base model. I then show how a second AI can use those processed outputs to predict on live data as well and superimpose its best estimate over the original image. This can be used further to isolate intersects between sets, boost the relevant values, and then reprocessed to generate the final letter. I hope this advances the efforts of the Vesuvius Challenge Team and increases the legibility of the letters by detecting ink where it should be present in the papyrus. 
