<!-- picture sources -->
#Stanford | Andrew Ng | Machine Learning
## Lecture Notes
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

###Example: Google News
Google news takes all sorts of web news, and clusters them together.  That way, the user gets a whole variety of different sources and articles grouped by a common topic.  

![google news clustering example][6]

As you can see, Google news's clustering algorithm find a group of news articles relating to the BP oil spill, and puts them together for the user to read.

###Example: DNA 

  [1]: https://lh5.googleusercontent.com/-xsTSv1cDLDc/U9BkOi7auKI/AAAAAAAAAJo/vp2mpLloPW4/s0/%25255BWeek%2525201%25255D%25255BSupervised%252520Learning%25255D%252520H
  [2]: https://lh6.googleusercontent.com/-AM3LqlGcF4g/U9BpYk_P54I/AAAAAAAAAKI/Zr_m8LviA8A/s0/%25255BWeek%2525201%25255D%25255BSupervised%252520Learning%25255D%252520Breast%252520cancer%252520prediction.png
  [3]: https://lh4.googleusercontent.com/-wgJnsnFYM_M/U9BrlnyVWMI/AAAAAAAAAKQ/tU85hYmCbAk/s0/%25255BWeek%2525202%25255D%25255BSupervised%252520Learning%25255D%252520Breast%252520cancer%252520prediction%25252C%2525202%252520dimensions.png
  [4]: https://lh6.googleusercontent.com/-ACvtRg3Yjt8/U9BuiA8hX6I/AAAAAAAAAKc/Yqec7ursfiQ/s0/%25255BWeek%2525201%25255D%25255BUnsupervised%252520Learning%25255D%252520Supervised%252520learning%252520example.png
  [5]: https://lh4.googleusercontent.com/-CY6Q_ZfOIO0/U9BvPi5R0BI/AAAAAAAAAKk/VC7gG0RB6jk/s0/%25255BWeek%2525201%25255D%25255BUnsupervised%252520Learning%25255D%252520Unsupervised%252520learning%252520example.png
  [6]: https://lh6.googleusercontent.com/-IY4Br9L5b08/U9Bw5VBcY2I/AAAAAAAAAKw/CDamiB31PLQ/s0/%25255BWeek%2525202%25255D%25255BUnsupervised%252520Learning%25255D%252520Google%252520news%252520clustering%252520example.png 