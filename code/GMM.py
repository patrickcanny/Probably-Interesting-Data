'''
    EECS 738
    Project 1: Probably Interesting Data
    File: GMM.py
    Implementation of Guassian Mixture Model
    Followed tutorials provided by python-course.eu: Gaussian Mixture Model
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, A, n_sources, iterations):
        self.iterations = iterations
        self.n_sources = n_sources
        self.A = A
        self.mu = None
        self.pi = None
        self.covariance = None
        self.xy = None

    def fit(self):
        self.reg_cov = 1e-6*np.identity(len(self.A[0]))
        x,y = np.meshgrid(np.sort(self.A[:,0]), np.sort(self.A[:,1]))
        self.xy = np.array([x.flatten(), y.flatten()]).T

        # set initial mu, covariance, and pi
        self.mu = np.random.randint(min(self.A[:,0]),max(self.A[:,0]),size=(self.n_sources,len(self.A[0])))
        self.covariance = np.zeros((self.n_sources,len(A[0]),len(A[0])))
        for dim in range(len(self.covariance)):
            np.fill_diagonal(self.covariance[dim],5)

        self.pi = np.ones(self.n_sources)/self.n_sources

        likelihoods = []

        self.plot_initial()

        for i in range(self.iterations):
            r_ic = np.zeros((len(self.A), len(self.covariance)))
            for m, co, p, r in zip(self.mu,self.covariance,self.pi, range(len(r_ic[0]))):
                co += self.reg_cov
                mn = multivariate_normal(mean = m, cov = co)
                r_ic[:,r] = p*mn.pdf(self.A)/np.sum([pi_c*multivariate_normal(mean=mu_c,cov=cov_c).pdf(A) for pi_c,mu_c,cov_c in zip(self.pi,self.mu,self.covariance+self.reg_cov)],axis=0)
            self.mu = []
            self.covariance = []
            self.pi = []
            for c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:,c],axis=0)
                mu_c = (1/m_c)*np.sum(self.A*r_ic[:,c].reshape(len(self.A),1),axis=0)
                self.mu.append(mu_c)
                self.covariance.append(((1/m_c)*np.dot((np.array(r_ic[:,c]).reshape(len(self.A),1)*(self.A-mu_c)).T,(self.A-mu_c)))+self.reg_cov)
                self.pi.append(m_c/np.sum(r_ic))


            likelihoods.append(np.log(np.sum([k*multivariate_normal(self.mu[i],self.covariance[j]).pdf(A) for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.covariance)))])))

        self.plot_likelihoods(likelihoods)

    def predict(self, B):
        fig3 = plt.figure(figsize=(10,10))
        ax2 = fig3.add_subplot(111)
        ax2.scatter(self.A[:,0],self.A[:,1])

        for m,c in zip(self.mu,self.covariance):
            multi_normal = multivariate_normal(mean=m, cov=c)
            ax2.contour(np.sort(self.A[:,0]),np.sort(self.A[:,1]),multi_normal.pdf(self.xy).reshape(len(self.A),len(self.A)),colors='black',alpha=0.3)
            ax2.scatter(m[0],m[1],c='grey',zorder=10,s=100)
            ax2.set_title('Final state')
            for y in B:
                ax2.scatter(y[0],y[1],c='orange',zorder=10,s=100)

        prediction = []
        for m,c in zip(self.mu,self.covariance):
            prediction.append(multivariate_normal(mean=m,cov=c).pdf(B)/np.sum([multivariate_normal(mean=mean,cov=cov).pdf(B) for mean,cov in zip(self.mu,self.covariance)]))

        plt.show()

    def plot_initial(self):
        fig = plt.figure(figsize=(10,10))
        ax0 = fig.add_subplot(111)
        ax0.scatter(self.A[:,0],self.A[:,1])
        ax0.set_title('Initial state')
        for m,c in zip(self.mu,self.covariance):
            c += self.reg_cov
            multi_normal = multivariate_normal(mean=m,cov=c)
            ax0.contour(np.sort(self.A[:,0]),np.sort(self.A[:,1]),multi_normal.pdf(self.xy).reshape(len(self.A),len(self.A)),colors='black',alpha=0.3)
            ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)

    def plot_likelihoods(self, likelihoods):
        fig2 = plt.figure(figsize=(10,10))
        ax1 = fig2.add_subplot(111)
        ax1.set_title('Log-Likelihood')
        ax1.plot(range(0,self.iterations,1),likelihoods)

