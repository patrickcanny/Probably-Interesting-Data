# Probably-Interesting-Data
EECS 738 - Machine Learning Project 1 Code

Submission by: Patrick Canny and Liam Ormiston

### Background & Inspiration
For this project, we selected the Red Wind Characteristics dataset, along with the Mushroom dataset. 

For the Red Wine Dataset, we formulated the following ideas:
- Predict Wine Quality 
  - K-Means where K = 2 or 3
    - Good, Average, Bad
- Categorize by similarity
  - K-Means Clustering
  - Gaussian Mixture Models

For the Mushroom Dataset, we formulated the following ideas: 
- Predict if a mushroom is poisonous
  - K-Means K = 2
    - Either edible or poisonous
  - Gaussian Mixture Model
- Categorize by similarity
  - K-Means Clustering
  - Gaussian Mixture Model

We will also consider adding in expectation-maximization into our implementations.

### Implementation
As we started working through our project, there were things that we adapted and changed.

To start, we greated our own Gaussian Mixture Model (GMM) by following a tutorial provided by python-course.eu: Gaussian Mixture Model.
This tutorial did a great job explaining what was going while creating the GMM. They provided figures and data examples as code was implemented.
Our GMM code can be found [here](/code/GMM.py)

As we started to test our model, we found that Jupyter Notbooks proved to be effective for continous testing and readability. You can find our Jupyter
notebooks in our [notebooks](/notebooks/) folder.

After we tested our GMM, we found that certain datasets worked better than others. This was due to a some attributes being catagorical instead of
continuous. 

We finally settled on the ***Iris dataset and the Mushroom dataset***. Please view those notebooks for our implementation.

Finally, we implemented a scikit learn library in order to test different covariances' accuracy versus our algorithm's accuracy. We compared the accuracy
of the scikit learn GMM with our GMM visually. With the accuracy precentage, we could look at our visualizations and see how close we were to their implementation.
See the gmm_covariances.ipynb in our [notebooks](/notebooks/) folder for the results.

