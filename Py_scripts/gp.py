import os
import numpy as np
import matplotlib.pyplot as plt
import json
import GPy
from sklearn.model_selection import train_test_split
import seaborn as sns

plt.rcParams["figure.figsize"] = [20, 10]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20


class LPI_GP:
    """Class implementing a Gaussian Process."""
    
    ## __init__
    #
    # The constructor
    # @param self : The object pointer
    # @param dir : Directory where data is stored (str)
    def __init__(self, input_file = None, output_file = None, var_file = None, train_frac = 0.4):
        """Class constructor."""
        self.input_file = input_file # file containg input data
        self.output_file = output_file # file containing output data
        self.var_file = var_file # file containing noise variance data
        self.train_frac = train_frac # set fraction for test-train split of data

    ## get_input
    #
    # Extracts the input data from json file and converts to numpy array
    # @param self : The object pointer
    def get_input(self):
        """Extracts the input data from json file and converts to numpy array."""
        # check input file exists
        os.path.exists(self.input_file)
        # open json file
        with open(self.input_file, 'r') as f:
            train_inputs = json.load(f)
        # place inputs into numpy array
        train_inputs = np.array(train_inputs)
        n = train_inputs.shape[0]
        input = np.zeros(n)
        for i in range(n):
            input[i] = train_inputs[i]
        return input

    ## get_noise_var
    #
    # Extracts the output variance data from json file and converts to numpy array
    # @param self : The object pointer
    def get_noise_var(self):
        """Extracts the output variance data from json file and converts to numpy array."""
        # check output variace file exists
        os.path.exists(self.var_file)
        # open json file
        with open(self.var_file, 'r') as f:
            train_outputs = json.load(f)
        # place output variances into numpy array
        train_outputs = np.array(train_outputs)
        n = train_outputs.shape[0]
        noise_var = np.zeros(n)
        for i in range(n):
            noise_var[i] = train_outputs[i]
        # convert to log as we train on the log output to avoid negative prediction from GP
        return np.log(noise_var)

    ## get_output
    #
    # Extracts the output data from json file and converts to numpy array
    # @param self : The object pointer
    def get_output(self):
        """Extracts the output data from json file and converts to numpy array."""
        # check output file exists
        os.path.exists(self.output_file)
        # open json file
        with open(self.output_file, 'r') as f:
            train_outputs = json.load(f)
        # place outputs into numpy array
        train_outputs = np.array(train_outputs)
        n = train_outputs.shape[0]
        output = np.zeros(n)
        for i in range(n):
            output[i] = train_outputs[i]
        return output
    
    ## set_training_data
    #
    # Performs test-train split of input, output and output variance, setting the training and test sets
    # @param self : The object pointer
    def set_training_data(self):
        """Performs test-train split of input, output and output variance, setting the training and test sets."""

        X = self.get_input() # inputs
        Y = self.get_output() # outputs
        noise = self.get_noise_var() # noise variances

        # split inputs and outputs into test-train sets
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = self.train_frac)

        # find indicies of the full which correspond to the test and train data points
        indxs_train = []
        indxs_test = []
        for i in range(len(X_train)):
            mask = np.in1d(X, X_train[i])
            indx = np.where(mask == True)[0][0]
            indxs_train.append(indx)
        for i in range(len(X_test)):
            mask = np.in1d(X, X_test[i])
            indx = np.where(mask == True)[0][0]
            indxs_test.append(indx)
        
        # use these indicies to extract the test and train noise variances
        noise_train = noise[indxs_train]
        noise_test = noise[indxs_test]

        # set the training and test data
        self.X_train = X_train[:,None]
        self.Y_train = Y_train[:,None]
        self.noise_train = noise_train
        self.X_test = X_test[:,None]
        self.Y_test = Y_test[:,None]
        self.noise_test = noise_test

        # data ranges for scaling hyper-parameters
        self.X_range = X.max()-X.min()
        self.Y_range = Y.max()-Y.min()
        self.noise_range = noise.max() - noise.min()
    
    ## update_noise_GP_kern
    #
    # Sets the noise variance model kernel (Exponential) for chosen hyper-paramters
    # @param self : The object pointer
    # @param l : Lengthscale hyper-parameter scale
    # @param var : Variance hyper-parameter scale
    def update_noise_GP_kern(self, l, var):
        """Sets the noise variance model kernel (Exponential) for chosen hyper-paramters."""
        l *= self.X_range # times by scale of input data
        var *= self.noise_range # times by scale of the output variances

        # kernel function using GPy package (Exponential kernel)
        self.kern_noise = GPy.kern.Exponential(input_dim=1, variance=var, lengthscale=l, ARD=True)
        self.K_noise = self.kern_noise.K(self.X_train, self.X_train)

    ## update_noise_GP_weights
    #
    # Sets the weights for the noise GP model. Weight(X) = [K(X,X) + 1e-6*I]^-1 y(X)
    # @param self : The object pointer
    # @param var_noise : Expected noise of output variance data (set to low value, as it is assumed the results are exact)
    def update_noise_GP_weights(self, var_noise = 1e-6):
        """Sets the weights for the noise GP model. Weight(X) = [K(X,X) + 1e-6*I]^-1 y(X)."""
        # use Cholesky decomposistion to avoid inverting matricies 
        self.L_noise = np.linalg.cholesky(self.K_noise + var_noise * self.noise_range* np.eye(len(self.X_train)))
        # estimates weights ([K+var*I] = LL^T, weight = [K+var*I]^-1y = L^T/(L/y))
        self.weights_noise = np.linalg.solve(self.L_noise.T, np.linalg.solve(self.L_noise, self.noise_train))
    
    ## update_noise_GP
    #
    # Updates the noise model kernel and weights for chosen hyper-paramters
    # @param self : The object pointer
    # @param l : Lengthscale hyper-parameter scale
    # @param var : Variance hyper-parameter scale
    def update_noise_gp(self, l, var):
        """Updates the noise model kernel and weights for chosen hyper-paramters."""
        self.update_noise_GP_kern(l, var)
        self.update_noise_GP_weights()

    ## get_noise_likelihood
    #
    # Finds the negative log-likelihood for the noise model at chosen hyper-paramters
    # @param self : The object pointer
    def get_noise_likelihood(self):
        """Finds the negative log-likelihood for the noise model at chosen hyper-paramters."""
        # output data (noise variances)
        y = self.noise_train
        # weights
        w = self.weights_noise
        # kernel - (K(X,X))
        K = self.K_noise
        K = np.array(K)
        # sign and log|K|
        sign, logdet = np.linalg.slogdet(K)
        n = len(K.diagonal())
        log_L = -0.5*np.dot(y.T, w) - 0.5*logdet - 0.5*n*np.log(2*np.pi)
        res = log_L
        return -1.0*res

    ## optimise_noise_GP
    #
    # Finds the optimal hyper-paramters (i.e minimises the negative log-likelihood) in the form of a simple grid search (Noise Model)
    # @param self : The object pointer
    def optimise_noise_GP(self):
        """Finds the optimal hyper-paramters (i.e minimises the negative log-likelihood) in the form of a 
           simple grid search (Noise Model)."""
        ells = np.geomspace(0.1, 10, 100) # lengthscale values to search
        vars = np.geomspace(0.1, 10, 100) # variance values to search
        # log likelihood values
        self.log_L_noise = np.zeros((len(ells), len(vars)))
        # find log likelihood for each ell and var
        for i, l in enumerate(ells):
            for j, v in enumerate(vars):
                self.update_noise_gp(l = l, var = v)
                self.log_L_noise[i,j] = self.get_noise_likelihood()

        # indicies of optimal values
        idx = np.where(self.log_L_noise == np.array(self.log_L_noise).min())
        # store optimal hyper-paramter values
        self.l_opt_noise = ells[idx[0][0]]
        self.var_opt_noise = vars[idx[1][0]]
        print('l = ' , self.l_opt_noise, 'var = ' , self.var_opt_noise)
        # update noise GP for optimal ell and var
        self.update_noise_gp(l = self.l_opt_noise, var = self.var_opt_noise)

    ## noise_GP_predict
    #
    # Predicts the noise variance at selcted inputs (X_star) using noise GP model
    # @param self : The object pointer
    # @param X_star : New points we wish to attain outputs for
    # @param get_err : GP predicted error
    def noise_GP_predict(self, X_star, get_err = False):
        """Predicts the noise variance at selcted inputs (X_star) using noise GP model."""

        K_star_noise = self.kern_noise.K(X_star, X_star) # K(X*,X*)
        k_star_noise = self.kern_noise.K(self.X_train, X_star) # K(X,X*) = k*
        f_star_noise = np.dot(k_star_noise.T, self.weights_noise) # k*^T[K+1e-6*I]^-1 y
        f_star_noise = np.exp(f_star_noise) # convert back as we train on log values

        if get_err:
            # find variances K* - k*^T[K+1e-6*I]^-1 k*
            v_noise = np.linalg.solve(self.L_noise, k_star_noise)
            V_star_noise = K_star_noise - np.dot(v_noise.T, v_noise)
            # convert variance of log outputs to variance of non-log outputs
            V_noise = f_star_noise**2 * np.diag(V_star_noise)
            # error = 2 * standard deviation
            err = 2.0*np.sqrt(V_noise)
            return f_star_noise, err
        else:
            return f_star_noise

    ## update_GP_kern
    #
    # Sets the GP model kernel (Rational Quadratic) for chosen hyper-paramters
    # @param self : The object pointer
    # @param l : Lengthscale hyper-parameter scale
    # @param var : Variance hyper-parameter scale
    def update_GP_kern(self, l, var):
        """Sets the GP model kernel (Rational Quadratic) for chosen hyper-paramters."""
        l *= self.X_range # scale lengthscale by input range
        var *= self.Y_range # scale variance by output range
        self.noise_var = self.noise_GP_predict(X_star=self.X_train) # get the noise variance for each input
        self.noise_cov = np.diag(self.noise_var) # convert to diagnonal matrix with the same shape as K(X,X)
        self.kern = GPy.kern.RatQuad(input_dim=1, variance=var, lengthscale=l, ARD = True) # Rational Quadratic kernel
        self.K = self.kern.K(self.X_train, self.X_train) # K(X,X)
        self.K += self.noise_cov # K(X,X) + D , D = Diag(var_noise(X))

    ## update_GP_weights
    #
    # Sets the weights for the GP model. Weight(X) = [K(X,X) + var(X)*I]^-1 y(X)
    # @param self : The object pointer
    def update_GP_weights(self):
        """Sets the weights for the GP model. Weight(X) = [K(X,X) + var(X)*I]^-1 y(X)."""
        # use Cholesky decomposistion to avoid inverting matricies
        # (added extra term to self.K for numerical stability)
        self.L = np.linalg.cholesky(self.K + 1e-6 * self.Y_range * np.eye(len(self.X_train)))
        # estimates weights ([K+var*I] = LL^T, weight = [K+var*I]^-1y = L^T/(L/y))
        self.weights = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y_train))

    ## update_GP
    #
    # Updates the GP model kernel and weights for chosen hyper-paramters
    # @param self : The object pointer
    # @param l : Lengthscale hyper-parameter scale
    # @param var : Variance hyper-parameter scale
    def update_GP(self, l, var):
        """Updates the GP model kernel and weights for chosen hyper-paramters."""
        self.update_GP_kern(l, var)
        self.update_GP_weights()

    ## get_GP_likelihood
    #
    # Finds the negative log-likelihood for the GP model at chosen hyper-paramters
    # @param self : The object pointer
    def get_GP_likelihood(self):
        """Finds the negative log-likelihood for the GP model at chosen hyper-paramters."""
        # output data
        y = self.Y_train
        # weights
        w = self.weights
        # kernel - K(X,X)
        K = self.K
        K = np.array(K)

        # sign and log|K|
        sign, logdet = np.linalg.slogdet(K)
        n = len(K.diagonal())
        log_L = -0.5*np.dot(y.T, w) - 0.5*logdet - 0.5*n*np.log(2*np.pi)
        res = log_L
        return -1.0*res

    ## optimise_noise_GP
    #
    # Finds the optimal hyper-paramters (i.e minimises the negative log-likelihood) in the form of a simple grid search (Noise Model)
    # @param self : The object pointer
    def optimise_GP(self):
        """Finds the optimal hyper-paramters (i.e minimises the negative log-likelihood) in the form of a 
           simple grid search (GP Model)."""
        ells = np.geomspace(0.1, 10, 100) # lengthscale values to search
        vars = np.geomspace(0.1, 10, 100) # variance values to search
        # find log likelihood for each ell var
        self.log_L = np.zeros((len(ells), len(vars)))
        for i, l in enumerate(ells):
            for j, v in enumerate(vars):
                self.update_GP(l = l, var = v)
                self.log_L[i,j] = self.get_GP_likelihood()

        # indicies of optimal values
        idx = np.where(self.log_L == np.array(self.log_L).min())
        # store optimal hyper-parameter values
        self.l_opt = ells[idx[0][0]]
        self.var_opt = vars[idx[1][0]]
        print('l = ', ells[idx[0][0]], 'var = ', vars[idx[1][0]])
        # update GP for optimal ell and var
        self.update_GP(l = self.l_opt, var = self.var_opt)
    
    ## GP_predict
    #
    # Predicts the noise variance at selcted inputs (X_star) using noise GP model
    # @param self : The object pointer
    # @param X_star : New points we wish to attain outputs for
    # @param get_var : GP predicted variance
    def GP_predict(self, X_star, get_var = False):
        """Predicts the output at selcted inputs (X_star) using GP model."""

        X_star = np.log(X_star) # need log of inputs as model is trained on log of intensity
        K_star = self.kern.K(X_star, X_star) # K(X*,X*)
        k_star = self.kern.K(self.X_train, X_star) # K(X,X*) = k*

        f_star = np.dot(k_star.T, self.weights) # k*^T[K(X,X) + var(X)*I]^-1 y
        if get_var:
            # get predicted noise variance at each new point
            self.noise_var_star = self.noise_GP_predict(X_star)
            self.noise_cov_star = np.diag(self.noise_var_star.flatten())

            # find epistemic varinace i.e K* - k*^T[K(X,X) + var(X)*I]^-1 k*
            v = np.linalg.solve(self.L, k_star)
            V_star_epi = K_star - np.dot(v.T, v)

            # get noise variance
            V_star_noise = self.noise_cov_star
    
            f_star = np.exp(f_star.flatten()) # require exponential as model is trained on log reflectivity
            V_epi = f_star**2 * np.diag(V_star_epi) # epistemic varinace for refelctivity
            V_noise = f_star**2 * np.diag(V_star_noise) # noise varinace for refelctivity
    
            return f_star.flatten(), V_epi.flatten(), V_noise.flatten()
        else:
            return np.exp(f_star).flatten()


    ## test_train_plot
    #
    # Produces test train plots (scatter and kde plots)
    # @param self : The object pointer
    def test_train_plot(self):
        """Produces test train plots (scatter and kde plots)."""

        # target data - all output data
        target_value = np.exp(self.get_output()).flatten()
        # output - training set
        Y_train = np.exp(self.Y_train).flatten()
        # output - test set
        Y_test = np.exp(self.Y_test).flatten()

        # predictions from training and test sets
        y_train_predict, var_train_epi, var_train_noise = self.GP_predict(X_star=np.exp(self.X_train), get_var = True)
        y_test_predict, var_test_epi, var_test_noise = self.GP_predict(X_star=np.exp(self.X_test), get_var = True)
        
        # root-mean-square error for training and test outputs
        rmse_train = np.sqrt(np.mean((Y_train-y_train_predict)**2))
        rmse_test = np.sqrt(np.mean((Y_test-y_test_predict)**2)) 
  
        # standard deviation for training and test outputs
        S_ptrain = np.sqrt(var_train_epi+var_train_noise)
        S_ptest = np.sqrt(var_test_epi+var_test_noise)

        # main figure panel
        fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)

        # kernel distribution estimation plot
        ax3 = sns.kdeplot(target_value, label = 'Target Data', linestyle='dashdot', linewidth = 5, color = 'black')
        ax3 = sns.kdeplot(y_train_predict, label=f'Train', color = 'blue')
        ax3 = sns.kdeplot(y_test_predict, label=f'Test', color = 'orange')
        ax3.set_xlabel(r'$\mathcal{P}$')

        # test-train output scatter plot
        ax1.scatter(Y_train, y_train_predict, label=f'Train (RSME = {np.round(rmse_train, 3)})', color = 'blue')
        ax1.plot([target_value.min(), target_value.max()], [target_value.min(), target_value.max()], 'k:', label = 'Target')
        ax1.set_xlabel(r'True Value - $\mathcal{P}$')
        ax1.set_ylabel(r'Predicted Value - $\mathcal{P}$')
        ax1.scatter(Y_test, y_test_predict, label=f'Test (RSME = {np.round(rmse_test, 3)})', color = 'orange')
        ax1.legend()

        # test-train output error scatter plot
        ax2.plot(abs(y_train_predict - Y_train), S_ptrain, 'o', label='Train', color = 'blue')
        ax2.plot([0, S_ptrain.max()], [0, S_ptrain.max()], 'k:', label = 'Target')
        ax2.plot(abs(y_test_predict - Y_test), S_ptest, 'o', label='Test', color='orange')
        ax2.plot([0, S_ptest.max()], [0, S_ptest.max()], 'k:', label = 'Target')
        ax2.set_xlabel(r'True Error - $\mathcal{P}$')
        ax2.set_ylabel(r'Predicted Error - $\mathcal{P}$')

        # scale subplots tp look tidy
        plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.5, 
                            hspace=0.4)

        plt.show()
