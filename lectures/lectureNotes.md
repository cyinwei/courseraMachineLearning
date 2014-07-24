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
insert price vs size

There are different ways to fit the data.  Do we choose a linear fit?  Quadratic fit?  Or higher exponents?  When we fit 

So supervised learning is when the _right answers_ are given.  This is also called **regression problem**.

>**Regression**: predicting a continuous output value.
>**Supervised learning**: where the right answers are given


####Example: Breast cancer detection

This is a classification problem.  




  
  