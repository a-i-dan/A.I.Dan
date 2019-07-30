---
layout:     post
title:      "A Brief Introduction to Privacy in Deep Learning"
date:       2019-03-18 12:00:00
author:     "A.I. Dan"
---

# A Brief Introduction to Privacy in Deep Learning

Successful state-of-the-art deep learning techniques require a massive collection of centralized training data, or data kept on one machine or location. While deep learning has shown unprecedented accuracy and success in a numerous amount of tasks, the common use of centralized training data restricts deep learning's applicability to fields where exposed data does not present privacy risks. This causes some fields such as healthcare to be limited in its benefits from deep learning.  

It is a widely known fact that to receive the best results from a deep learning model, the training datasets must be bigger with lots of variety. The more data to learn from, the better the results. A model that lacks access to larger datasets and variety will commonly be subject to overfitting and poor end results.      

In the field of medicine, deep learning has proven to be potentially life saving and vital to the field's progression. The problem arises with the highly private and sensitive information that medical institutions carry. Due to the high privacy expectations and regulations, the data cannot be legally shared with other institutions. Each institution is then forced to train models on only the data available to them: the patient data they own and not from other institutions. This results in models producing results that do not generalize and will perform poorly when presented with new data.

## Federated Learning

The goal of federated learning algorithms is to train a model on data kept among a large number of different participating devices, called <b>clients</b>. As mentioned earlier, training deep learning models requires data to be stored locally along with the model itself. In other words, the training data is brought to the deep learning model. Federated learning aims to bring the model to the training data.

The model will use a large number of clients to pull from. Training data will still be stored locally on each device, but each device will download the model, perform computations on their local data, produce a small update for the model, then update the model and send it back to the central server. This enables each client to contribute to the global models training while still storing its data locally. Each client will only send back an update to the global model, where it will then be averaged with other clients updates.

<img src='a-i-dan.github.io/images/privacy in deep learning figures.png?raw=true' style='margin: auto; display: block;'>

<b>Recommended Read:</b>

* [Federated Learning: Strategies For Improving Communication Efficiency](https://arxiv.org/pdf/1610.05492.pdf)


## Secure Multi-Party Computation

Secure multi-party computation (MPC) aims to allow multiple parties to perform a computation over their input data while ensuring the data is kept private and the final result is accurate.

Imagine there are five friends sitting at a table. They want to know which one of them has the highest salary, but all five of them do not want others to know their own salary unless they have the highest. An easy way to solve this problem would be to bring in a trustworthy third party, where each of the five friends can tell this third party their salaries and the third party will announce who has the highest salary without ever giving out information about anyone else's salary. Unfortunately, such a trusted third party is hard to come by. Thus, MPC aims to replace this needed trustworthy third party with a cryptographic protocol that still ensures data privacy.  

MPC is a growing area of cryptography within machine learning and deep learning. Below are some other resources and recommended reads:

* [Secure Multiparty Computation For Privacy-Preserving Data Mining](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.215.6337&rep=rep1&type=pdf)

* [Is Multiparty Computation Any Good In Practice?](http://cs.au.dk/~orlandi/icassp-draft.pdf)


## Differential Privacy

An algorithm that analyzes data can be defined as differentially private if the output cannot be used to determine if an individual's data was used in the original dataset that the computations were performed on. The goal of differential privacy is to reduce the risk of an individual's information being compromised by having computations run over the data not be fully reliant on one individual's information.

This is done by adding enough noise to the dataset to impact the data in a way that no specific details about an individual can be determined. To ensure the same level of anonymity for each individual, as a dataset decreases in size, more noise is added. As a dataset grows, less noise needs to be used to achieve the same resulting privacy for each individual.

As an example, for a dataset containing information from two individuals, each individual makes up 50% of the dataset. A larger amount of noise will need to be added to ensure that both individuals data is kept private. A dataset with one million people will have each person make up 0.000001% of the dataset. Thus, less noise is needed to mask the individual's data.

<b>Recommended Reads:</b>

* [Privacy-Preserving Datamining on Vertically Partitioned Databases](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/crypto04-dn.pdf)

* [Differential Privacy](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/dwork.pdf)

* [Revealing Information while Preserving Privacy](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.1298&rep=rep1&type=pdf)
