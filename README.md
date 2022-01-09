## Project Name & Description

This is the underlying source code behind the AI prediction system used for the Brown Critical Review. The Brown Critical Review is a university-sanctioned organization that publishes course reviews on every class that has been offered in the last 20 years at Brown University. 

A major problem that the organization suffers from is incomplete data. Many times one of the most crucial datapoints "Average hours of work per week" is left empty, and this is the most sought after datapoint by students. This project aims to solve this problem by developing a neural network based regression system that predicts this datapoint based off of other datapoints that have been filled out by students. 

The below sourcecode contains a series of different models that were tested to solve this problem. The first model  (RNN) is a basic recurrent neural network that takes as input 3 different text fields, and uses gated recurrent networks, dense layers, MSE loss function, and RELU activation to output predictions. 

The second model (RNN_DROPOUT) is a identical to the RNN model but with the addition of a dropout layer. 

The final model (which is the model that was eventually used in production) uses both text inputs and a set of 15 additional numerical inputs (which can be analyzed in detail in the following file)
```
2020-2019 Numerical Review Data.csv
```

along with GRUs and a series of dense layers to create predictions. This model was the most accurate, and led to an average error of only 3.8 hours per week on the test dataset. 

## Project Status

This project is completed

## Project Screen Shot(s)

#### Example:   

To see the project in action, please visit 

https://thecriticalreview.org/

A screenshot of the program output is below

![ScreenShot](https://github.com/singhru27/Course-Hours-Prediction/blob/master/screenshots/Default.png)


## Installation and Setup Instructions

To run the program, first make sure that Python is installed. Then, install all dependencies as delineated in the 

```
requirements.txt
```
file. To run the basic RNN model and print out training + test accuracy, run the following:

```
main.py RNN
```

To run the modified RNN model with an additional dropout layer, run the following:

```
main.py RNN_DROPOUT
```

To run the production model which utilizes both RNN layers as well as dense layers for the numerical input (this model is what was eventually used for the website itself), run the following:

```
main.py RNN_ALL
```



