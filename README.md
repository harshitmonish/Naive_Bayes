# Project Title:
## Given a movie review, predict the rating given by the reviewer.(Sentiment Analysis)

### In this problem, we will use the Naive Bayes algorithm for text classification. The dataset for this problem is a subset of the IMDB movie review dataset and has been obtained from [this_website](http://ai.stanford.edu/~amaas/data/sentiment/).A review comes from one of the eight categories (class label).Here, class label represents rating given by the user along with the review. Dataset is provided in four files i) Train text ii) Train labels iii) Test text iv) Test labels. Text files contain one review in each line and label files contain the corresponding rating.

* Have used cpickle module to dump the trained and test model objects.
* Will perform stemming and remove the stopwords in the training as well as the test data. Stemming script is inside the script folder
* Will use the Laplace smoothing for Naive Bayes to avoid any zero probabilities. c = 1.
* Will report the test set accuracy that is obtained by randomly guessing one of the categories as the target class for each of the articles (random prediction). Also reported the accuracy obtained by simply predicting the class which occurs most of the times in the training data (majority prediction) and Naive Bayes prediction.
* Will draw the confusion matrix for the results.


