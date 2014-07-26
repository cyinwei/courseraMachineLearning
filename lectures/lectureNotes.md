<!-- picture sources -->
#Stanford | Andrew Ng | Machine Learning
#Lecture Notes
Motivation:  For Will Johnson's peak detection algorithm on Interspec.

## I. Introduction (Week 1)
----

###Welcome
Machine learning is really prevalent.

- Google search
- Facebook tagging
- Spam email filtering

For Andrew, the motivation is creating the AI Dream, where `computerIntelligence >= humanIntelligence`.  

In this class we learn:

1. About machine learning algorithms, the math and concepts 
2. How to implement them in practical cases (that's where they're useful)
3. _To really understand how they work from the ground up_

####Why is machine learning so prevalent?  Where is it prevalent?  

Machine learning grew out of work in AI.  We could find simple algorithms (Dijkstras), but it was difficult to develop general case algorithms for difficult, general problems like pattern recognition.  We use machine learning to help solve these problems that we can't program by hand, like:
 
- Autonomous helicopter driving, handwriting recognition, most of Natural Language Processing (NLP), Computer Vision.
 - Self-customizing programs
  - Amazon, Netflix product recommendations
- Understanding human learning (brain, real AI)
- Database Mining
  - We have a large set of data to analyze with the web, smartphones, computation biology, engineering

----
###What is machine learning?

####What is machine learning, and when do we use it?
There isn't a well accepted definition among experts of what exactly is machine learning.
Machine Learning definitions:

Arthur Samuel (1959)'s definition of machine learning
:   Field of study that gives computers the ability to learn without being explicitly

- Arthur's claim to fame was that he designed a checker program that played and learned from itself (tens of thousands of times).  It became a better checkers player than Arthur himself.

Tom Mitchell (1998)'s Well-posed Learning Problem
:   A computer program is said to _learn_ from experience E, with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.  

- For the checker program:
  - T = checkers
  - P = probability of winning vs. a random opponent (randomized skill ?: equal skill)
  - E = playing checker games versus itself
- Another example: email program spam filtering
  - T = classifying emails as spam or not spam
  - P = the percentage of emails correctly classified as spam or not spam
  - E = watching the user label emails and spam or not

####The two main types of machine learning

Supervised learning
:   We teach the computer what to do

Unsupervised learning
:   We let the computer learn it by itself

Other terms, but we spend most of our time on the two types above

- Reinforcement learning, recommender systems.

**We also emphasize practical advice for applying learning algorithms.**  This is equally, if not more important than learning about learning algorithms.  Think of these algorithms as a set of tools.  And lets make the carpenter and his tools analogy, we need to first get the tools, and more importantly, _learn how to use them_.

----

###Supervised Learning
We start with am example, then get to the definition (deductive reasoning).

####Example: Housing price prediction
![price vs size of a house][1]

There are different ways to fit the data.  Do we choose a linear fit?  Quadratic fit?  Or higher exponents?  When we fit 

So supervised learning is when the _right answers_ are given.  This is also called **regression problem**, where our algorithm tries to give a continuous output..

####Example: Breast cancer detection
![breast cancer prediction][2]

This is a **classification problem**, where we try and classify, or map the output to a discrete value.

Of course, for both classification and regression, our algorithm, or function can take in more than one dimension.  Generally, the more dimension, feature, or attributes, the more accurate our algorithm will be.  See below.  For some problems, we would want an infinite number of dimensions for our algorithm.  We can use something called a _support vector machine_ that uses a neat math trick (something like storing data in higher dimensions to make them curvy?) that can do that.

![breast cancer prediction in 2d and more][3]

**Terms**

>**Supervised learning**
>:  Category of machine learning where the answers for each data point are given, and the machine creates an algorithm to predict certain output for a given input.

>   **Regression**
>       :  Supervised learning predicting a **_continuous_** output value.

>   **Classification**
>       :  Supervised learning predicting a **_discrete_** output value.

----
##Unsupervised Learning
We talked about supervised learning, where each data point is labeled, and told explicitly what is "right" or not.  It looks like this.

![example of supervised learning][4]

As you can see, the data is marked Os and Xs; they are labeled; an answer is already given for the data.  This is what unsupervised learning is.

![example of unsupervised learning][5]

Here, the data is not labeled, we don't give the machine any answers at all.  We're not told what do to with the data.  So with unsupervised learning, we're given _unbiased data_, and the machine will try and find **some structure, some pattern** to the data.  The example above is a **clustering algorithm**, as the machine will group the data into two big clusters.

###Clustering
####Example: Google News
Google news takes all sorts of web news, and clusters them together.  That way, the user gets a whole variety of different sources and articles grouped by a common topic.  

![google news clustering example][6]

As you can see, Google news's clustering algorithm find a group of news articles relating to the BP oil spill, and puts them together for the user to read.

####Example: DNA 
Clustering is also useful for DNA analysis.

![DNA clustering example][7]
Source: Su-In Lee, Dana Pe'er, Aimee Dudley, George Church, Daphne Koller

We don't know anything about the structure of the genes, so we use a clustering algorithm to group and categorize and find structure for the individuals, as shown by the colors.

####Other Examples
![Other clustering examples][8]

We can use clustering algorithms to better understand, or optimize other problems with a complex amount of data.

- With computing servers, we can use a clustering algorithm to put servers who communicate together to minimize physical distance and decrease lag.
- With social network analysis, we can cluster people who _talk_ to each other (Facebook, Twitter, email) to create social groups that allows us to better understand the people involved.
- With market segmentation, we can cluster specific groups to better target them.
- In astrophysics, we can cluster those objects together by features to find relationships between them (not really sure).

All these are examples of the uses of the clustering algorithm, which is just **_one_** type of unsupervised learning.  Another example is the **cocktail party problem**.

###Cocktail party problem
In a party, we have two different hearing inputs (2 different microphones) at different locations.  Let's say we have two different speakers, talking.  How do we separate the people talking?  Note we have to specify how many audio sources we have for the algorithm to work.

![cocktail party problem][9]

####Example: two people counting to ten, one in spanish, one in english
With the algorithm, we can clearly hear the english and spanish voices separated in two different outputs.

####Example: radio music and a person counting to ten in english
With the algorithm, we can almost fully get the music, almost _almost_ filter out the music on the second output.

Source for audio: Te-Won Lee

####The algorithm itself
Surprisingly, the algorithm is actually just one line of `Octave` code:
```matlab
[W,s,v] = svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x');
```
Source: Sam Roweis, Yair Weiss & Eero Simoncelli

Note that `svd` and `repmat` are built in Octave linear algebra functions.

###Why choose Octave?
We learn much faster with Octave because we don't have link lots of C++ or Java libraries to find these linear algebra functions.

In real life, we often prototype the functions in an "easy" language like Matlab or Octave, then implement the algorithm in C++ or Java for speed and wrapping purposes.

Andrew says that our learning time is one of the most valuable resources a student has (touché), so if we learn faster, and are more productive with Octave, then why not use it?

----
##II. Linear Regression with One Variable (Week 1)
----
###Model Representation 
Linear regression is a type of regression, which is a type of **supervised learning.**  
####Introduction by Example: Housing prices in Portland
![portland housing price linear regression][10]

Linear regression is fitting a straight (linear) line that "best fits" the data.  That way we can predict a (continuous) value for a given input.  In this case, our input is 1 variable, which is the size in squared feet.  Our output is also 1 variable, which is the price in k$.  

This fits in with supervised machine learning, because we give the machine some set of training data, let the machine train itself (in this case, by defining a fit function), and now expect the machine to give answer for *any* input.

#####Notation
One way of 


  [1]: https://lh5.googleusercontent.com/-xsTSv1cDLDc/U9BkOi7auKI/AAAAAAAAAJo/vp2mpLloPW4/s0/%25255BWeek%2525201%25255D%25255BSupervised%252520Learning%25255D%252520H
  [2]: https://lh6.googleusercontent.com/-AM3LqlGcF4g/U9BpYk_P54I/AAAAAAAAAKI/Zr_m8LviA8A/s0/%25255BWeek%2525201%25255D%25255BSupervised%252520Learning%25255D%252520Breast%252520cancer%252520prediction.png
  [3]: https://lh4.googleusercontent.com/-wgJnsnFYM_M/U9BrlnyVWMI/AAAAAAAAAKQ/tU85hYmCbAk/s0/%25255BWeek%2525202%25255D%25255BSupervised%252520Learning%25255D%252520Breast%252520cancer%252520prediction%25252C%2525202%252520dimensions.png
  [4]: https://lh6.googleusercontent.com/-ACvtRg3Yjt8/U9BuiA8hX6I/AAAAAAAAAKc/Yqec7ursfiQ/s0/%25255BWeek%2525201%25255D%25255BUnsupervised%252520Learning%25255D%252520Supervised%252520learning%252520example.png
  [5]: https://lh4.googleusercontent.com/-CY6Q_ZfOIO0/U9BvPi5R0BI/AAAAAAAAAKk/VC7gG0RB6jk/s0/%25255BWeek%2525201%25255D%25255BUnsupervised%252520Learning%25255D%252520Unsupervised%252520learning%252520example.png
  [6]: https://lh6.googleusercontent.com/-IY4Br9L5b08/U9Bw5VBcY2I/AAAAAAAAAKw/CDamiB31PLQ/s0/%25255BWeek%2525202%25255D%25255BUnsupervised%252520Learning%25255D%252520Google%252520news%252520clustering%252520example.png
  [7]: https://lh6.googleusercontent.com/-ph8MgNfy-4Y/U9LTu1XUk8I/AAAAAAAAALA/RWJVsQAv5kM/s0/%25255BWeek%2525202%25255D%25255BUnsupervised%252520Learning%25255D%252520DNA%252520clustering.png
  [8]: https://lh5.googleusercontent.com/-jVczsrla9Ro/U9LU2Iqqo7I/AAAAAAAAALI/eMfPWTeYP38/s0/%25255BWeek%2525202%25255D%25255BUnsupervised%252520Learning%25255D%252520Other%252520clustering%252520examples.png
  [9]: https://lh4.googleusercontent.com/-d4Smy59LzKA/U9LW-kJ_hmI/AAAAAAAAALU/qMlWPnP0phY/s0/%25255BWeek%2525202%25255D%25255BUnsupervised%252520Learning%25255D%252520Cocktail%252520party%252520problem.png
  [10]: https://lh6.googleusercontent.com/-gSODDORlRdU/U9LqlbJuVkI/AAAAAAAAALk/VzL4mEl_64g/s0/%25255B2%25255D%25255BModel%252520Representation%25255D%252520Housing%252520price%252520example%252520linear%252520regression.png 