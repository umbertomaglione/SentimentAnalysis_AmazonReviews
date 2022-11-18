# An introduction to Sentiment Analysis with Python

## What is Sentiment Analysis?

"Sentiment analysis is a natural language processing technique use to determine whether data is positive, negative or neutral".
It is useful to sort data at scale, get access to real-time informations, and avoiding biases when determining the overall sentiment.

There are three types of algorithms:
+ Rule-based
+ Automatic
+ Hybrid (a combination, usually more accurate)

#### Rule-based
Also called lexical approach, it involves computational linguistics (some examples: stemming, tokenization, part-of-speech, tagging, parsing, lexicons). This approach first defines two lists of vocabs/expressions, and then count the number of positive and negative that appear in a given text.

It's a very limited method:
+ it doesn't take into account how words are combined
+ thus, requires fine-tuning based on new rules (which means a regular expenditure)
+ slow to train

#### Automatic
Relies on machine learning techniques, and is modeled as a classification problem.
In the training process, text is extracted into a vector and the model learns to associate the input to the associated tag (this means it's supervised learning).

The main challenges of sentiment analysis are presented by:
+ negations
+ sarcasm
+ neutral, objective comments
+ idiomatic expressions, new terms, emojis
+ different languages



## VADER Sentiment Analysis

VADER (Valence Aware Dictionary for sEntiment Reasoning) represents a lexical approach. The model relies on a dictionary of sentiment and assesses the overall sentiment by looking at the valence of the individual words.
In my work, I use the SentimentIntensityAnalyzer from nltk, which gives a score respectively to the negative, neutral, and positive sentiment.

### Bag Of Words
The model first transforms each sentence into a vector, where every element indexes how many times a certain word is repeated. In this process, the sentences are stripped away by their punctuation and the context is completely disregarded. This process is called tokenization. Sometimes, bi-grams (or, more in general, n-grams) are used to keep track of the frequency of a pair of words (or n words), rather than each individual word.



## BERT model

The BERT model uses positional encoding: before feeding each word to the neural network, they are labelled by their order in the phrase. In theory, the model would interpret the meaning of a sentence based on the positioning on the words.

Another fundamental principle is "attention". As the BERT model was originally designed for translation, it interprets every word in relation to others. For example, in latin languages adjective change based on the gender of words, meaning that a correct model has to look at the relative noun when traslating.
The model learns which word it has to "attend" through repetition. In particular, the model has "self attention". This allows to disambiguate homonymous words by considering the context.



## Sci-kit learn models


### Support Vector Machines (SVM)

SVM creates an optimal hyperplane, maximizing a maximum marginal hyperplane (MMH).
Generalization of Optimally Separating Hyperplane (allows for overlapping groups).

Hyperplane: decision plane to separate data points into classes
Support vectors: closest observations to the hyperplane
Margin: perpendicular distance from the line to the support vectors (bigger gaps represent more segregated classes).

The hyperplane is chosen by an iterative process minimizing the (mis)classification error.

Kernel tricks allow to handle nonlinear input spaces (see polynomial and RBF kernels)
C parameter: penalty on error term, trade-off between decision boundary and misclassification term (as it increases, so does the margin)
Gamma: measure of fitting (low: considers only nearby points, high: over-fitting)

Pros: 
+ faster than Naïve Bayes
+ uses less memory because trains on a subset
+ works well with high dimensions

Cons:
+ slower than Naïve Bayes for large datasets
+ works poorly with overlapping classes
+ sensitive to the type of kernel

Differences with LDA:
+ no assumptions on distribution
+ optimization problem (LDA has analytical solution)
+ allows for non-linear classifications
+ uses subset


## Logistic Regression

The logistic regression is a statistical method for predicting classes, and it does so by computing the the probability of an event occurrence. The model can be estimated by MLE or Least Squares.
In our case, the regression is multinomial, as there are more than two categories.

The simplicity of this model represents a strength, since it's easy to implement and interpret.
Its main weakness lies in the fact that it doesn't handle well a large number of features. It is vulnerable to overfitting and relies on a strong correlation between target and indepent variables (the latter preferably weakly uncorrelated).


## Decision Tree

A decision tree can be represented as a flow-diagram with a tree structure, where a node represents feature, the branch represents a decision rule, and each leaf node represents the outcome. This makes decision trees easy to understand, as it shows the internal decision-making (i.e. white box algorithm).

Feature selection is done by Attribute Selection Measures (ASM). ASM assigns a rank to every feature based on how well it explains the dataset, and splits the dataset by classifying the data points based on the highest scoring attribute.
This procedure is repeated and at aech further step the dataset is divided into a smaller sample.
The process ends when there are no more remaining attributes.

This model is particularly versatile, since it can easilly capture non-linear patterns without distributional assumptions (it's a "non-parametric algorithm"), but it is sensitive to unbalanced and noisy data. In particular, small variation in data can result in defferent decisions.


<div style="page-break-after: always;"></div>


# My results

![ScreenShot](https://github.com/umbertomaglione/SentimentAnalysis_AmazonReviews/blob/main/vader_barplot.png "Vader Barplot")

As expected, the results from VADER match the observed scores.

![ScreenShot](https://github.com/umbertomaglione/SentimentAnalysis_AmazonReviews/blob/main/pairplot.png "Vader-Roberta Pairplot")

The pairplot doesn't show any anomalies. When confronted to the VADER scores, Roberta is more conservative when using the Neutral label.

Furthermore, I analyzed some cases in which the models score was particularly high, whereas the rating was low (and viceversa).

The highest positive-scoring 1-star reviews are, respectively for VADER and Roberta:
+ 'I felt energized within five minutes, but it lasted for about 45 minutes. I paid $3.99 for this drink. I could have just drunk a cup of coffee and saved my money.'
+ 'So we cancelled the order.  It was cancelled without any problem.  That is a positive note...'
In the first case, VADER score is probably mislead by "energized" and the lack of explicitly negative words.
In the case of Roberta, the model doesn't catch the irony inteded by the author, but it's a justifiable error.

The highest negative-scoring 5-stars review is for both model: 'this was sooooo deliscious but too bad i ate em too fast and gained 2 pds! my fault'. In this case, the misunderstanding is excusable too.

I proceed by applying some arbitrarly chosen models from the sklearn library (the ones that I discussed above). These models are not comparable with the ones before, as they predict an underlying class instead of assigning scores for each feature.
To implement them, I first split the dataset into a training and a test subset, in order to cross-validate. When vectorizing, I apply the TF-IDF (term frequency–inverse document frequency) vectorizer, which assigns heavier weights to words that appear less often (assuming they are more informative).

The accuracy score shows the ratio between the number of correct predictions. The results are:
+ 0.79 for the Linear SVM
+ 0.79 for the Logistic Regression
+ 0.74 for the Decision Tree

The F-score is an harmonic mean of the model's precision and recall. The results scores are (respectively for the negative, neutral, and positive):
+ for the Linear SVM: 0, 0, 0.88
+ for the Logistic Regression: 0, 0, 0.88
+ for the Decision Tree: 0.24, 0.21, 0.84

We notice that the results are very disappointing for the negative and neutral features. This reflects a bias for 5-stars reviews present in our data, rather than models' imperfections.

![ScreenShot](https://github.com/umbertomaglione/SentimentAnalysis_AmazonReviews/blob/main/stars.png "Stars")

Furthermore, I manually balanced the dataset to account for this bias. The results significantly improve, also (or mainly) due to the larger sample-size. I also noticed a tradeoff between positive and negative accuracy for the Decision Tree.

I conclude my work by fine-tuning the Linear SVM. The Grid Search method iteratively splits the data in order to cross-validate some optimal parameters (in my case, the Kernel type and the C-value). Afterwards, I save my model with the pickle library, which will allow me to later perform sentiment analysis with the selected parameters.



