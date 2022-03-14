# CSLIproblingbook
Source files for my CSLI Probabilistic Approaches to Linguistic Theory chapter


When running the file probILM.py, you will be asked to input values for the model parameters:

## Parameters for the model

- Number of situations and predicates for the starting condition: since num_sits must be equal to num_preds, only one value is input
- Data size: the number of situation-predicate pairs each generation receives as learning data. 
- Delta: if a generation does not directly experience a descrption of a situation, they must infer which of the predicetes they have experienced to use to describe this situation. The delta parameter controls how distance between situations in the order affects this reasoning such that delta is a numeral that multiples the distance measure between situations
- Number of generations: the number of generations the iterated learning model will cycle through.
-  Noise: noise is modelled by applying a gaussian distribution to the learning data such that if the noise free message is [sit, pred], the learner has uncertainty about which sit is being described with sit as the mean and ```noise_var``` as the  standard deviation.
- File name: the csv file name under which the results for the final generation will be saved.

The model is also available as a Jupyter Lab notebook
