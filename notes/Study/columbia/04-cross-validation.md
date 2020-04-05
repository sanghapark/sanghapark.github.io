# Cross Validation

An easier way to evaluate the model is to use cross-validation.

The rocedure for $$K$$-fold cross-validation is very simple:

1. Randomly split the data into $$K$$ roughly equal groups.
2. Learn the model on $$K - 1$$ groups and predict the held-out $$K$$th group.
3. Do this $$K$$ times, holding out each group once
4. Evaluate performance using the cumulative set of predictions.

For the case of the regularization parameter $$\lambda$$, the above sequence can be run for several values with the best-performing value of $$\lambda$$ chosen.

>  The data you test the model on should never be used to train the model!