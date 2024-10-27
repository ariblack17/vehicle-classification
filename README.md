# vehicle-classification
A group project developed for UConn's graduate Big Data Analytics course, CSE 5717. 

## Project overview
Given a picture of a vehicle, the classification model developed in this project is able to predict its make and model with high accuracy and precision.

The input to the model is a clear image of a vehicle and the coordinates of the vehicle's bounding box within the image, and the output is the vehicle's year, make, and model.

The primary metric for evaluating the model's performance was to calculate the percentage of accurate classifications on an unseen testing set of images. The team also considered the relative proximity of the model's predictions for each class by evaluating precision, recall, and F1 score.

## The dataset
The Stanford Cars Dataset, containing 16,185 images of 196 classes of vehicles was selected for this project. Sourced from Jonathan Krause (Stanford, PhD) in his online archive, the dataset can be divided into two major components: the images and the annotations.

### Images
The images of the dataset are small JPGs with slight deviations in size, most closely cropped to their subjects. Few images were of relatively lower quality than the rest, but none were poor enough to warrant removal.

### Annotations
