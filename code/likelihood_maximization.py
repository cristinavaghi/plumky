from model_definition import Model
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.stats import t, f
import warnings
import numpy as np

def gradient_df( function, param_loc ):
    '''
    gradient_df computes the gradient of a function around the value param_loc
    Input:
    - function: lambda function
    - param_loc: value of the parameter where the gradient of the function is evaluated
    Output:
    - gradient of f around param_loc
    '''
    x           = list()
    P           = len(param_loc)
    param_minus = np.zeros(P)
    param_plus  = np.zeros(P)

    for j in range(P):
        x.append(param_loc[j])

    epsi = 1e-10
    h    = 1e-10
    df   = np.zeros(P)
    for i in range(P):
        xi = np.array([x[i]])
        if np.linalg.norm(xi) > epsi: # if xi !=0, use relative finite difference to scale at the scale of x
            xi_minus_un = np.concatenate((x[:i], xi*(1-h), x[i+1:]))
            xi_plus_un  = np.concatenate((x[:i], xi*(1+h), x[i+1:]))
            for j in range(P):
                param_minus[j] = xi_minus_un[j]
                param_plus[j]  = xi_plus_un[j]
            df_loc = (function(param_plus)- function(param_minus))/(2*xi[0]*h)
            df[i]  = df_loc
        else: # if xi == 0 (or very small), use classical finite differences with smaller h$
            h = h*epsi
            xi_minus_un = np.concatenate((x[:i], xi-h, x[i+1:]))
            xi_plus_un  = np.concatenate((x[:i], xi+h, x[i+1:]))
            for j in range(P):
                param_minus[j] = xi_minus_un[j]
                param_plus[j]  = xi_plus_un[j]
            df_loc = (function(param_plus)- function(param_minus))/(2*h)
            df[i]  = df_loc
    return df

def compute_jacobian(function, x, param_loc ):
    '''
    compute_jacobian computes the jacobian of a function around the value param_loc
    Input:
    - function: lambda function
    - time (numpy array): time values where the function is evaluated
    - param_loc (numpy array): value of the parameter where the gradient of the function is evaluated
    Output:
    - J (numpy array): jacobian of f
    '''
    P = len(param_loc) # length of the vector of parameters
    K = len(x) # length of the vector of the observations
    J = np.zeros([K,P]) # the jacobian is a numpy vector of dimension P x K
    for k in range (K):
        x_loc = x[k]
        J[k,:] = gradient_df( lambda p: (function(x_loc, p)), param_loc )
    return J

def sumOfSquaredError(function, xData, yData, parameterTuple):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    return np.sum((yData - function(xData, *parameterTuple)) ** 2)

def generate_initial_parameters(function, xData, yData, parameterBounds):
    # "seed" the numpy random number generator for repeatable results
    func = lambda parameterTuple: sumOfSquaredError(function, xData, yData, parameterTuple)
    result = differential_evolution(func, parameterBounds, seed=20)
    return result.x


def nonlinear_regression(model, xData, yData, sigma1, sigma2):
    '''
    Nonlinear regression with scipy.optimize.minimize
    Nelder-Mead algorithm to minimize the log likelihood function
    To guarantee positivity the logarithm of the parameters is computed.
    A proportional error model parameter is considered.

    yData ~ N(model(xData, p), (sigma1+sigma2*model(xData, p))^2)

    Input:
    - model (class Model): containing the function, initial guess and parameter bounds
    - xData, yData (numpy array of dimension n): vectors containing the x and y values
    - sigma1 (scalar): constant error model parameter.
    - sigma2 (scalar): proportional error model parameter.

    The results of nonlinear regression are stored in the class objects Model.
    '''
    try:
        K           = len(xData)
        C = lambda x, p: np.diag((sigma1+sigma2*model.function(x,*np.exp(p)))**2)
        c_inv = lambda x, p: 1/(sigma1+sigma2*model.function(x,*np.exp(p)))**2

        residual    = lambda y,x,p: 1/2*sum((y-model.function(x,*np.exp(p)))**2*c_inv(x,p))

        log_likelihood = lambda y, x, p: sum(1/2*np.log((2*np.pi)**K/c_inv(x,p))) + residual(y,x,p)
        r           = minimize(lambda p: log_likelihood(yData, xData, p), x0 = np.log(model.initial_guess), method = 'Nelder-Mead')

        model.params['Values'] = np.exp(r.x)
        P           = len(r.x)


        C_inv_theta = np.diag(c_inv(xData,r.x))
        NMSE        = 2*residual(yData,xData,r.x)/(K-P)
        J           = compute_jacobian(lambda x, p: model.function(x, *np.exp(p)), xData,  r.x)

        FIM = np.zeros((P,P)) # Fisher information matrix
        for i in range(P):
            dC_dtheta_i = 2*sigma2*(sigma1 + sigma2*np.diag(model.function(xData,*np.exp(r.x)))*J[:,i])
            for j in range(P):
                dC_dtheta_j = 2*sigma2*(sigma1 + sigma2*np.diag(model.function(xData,*np.exp(r.x)))*J[:,j])
                M_loc = np.matmul(np.matmul(np.matmul(C_inv_theta,dC_dtheta_i), C_inv_theta), dC_dtheta_j)
                FIM[i,j] = np.matmul(np.matmul(J[:,i], C_inv_theta), J[:,j]) + 1/2*np.trace(M_loc)
        # variance covariance matrix
        cov = NMSE*np.matmul(np.matmul(np.diag(np.exp(r.x)),np.linalg.inv(FIM)),np.diag(np.exp(r.x)))
        # standard errors
        st_err      = np.sqrt(np.diag(cov))
        st_err_norm = np.abs(st_err/np.exp(r.x)*100)
        model.params['StdErr']    = st_err
        model.params['StdErr[%]'] = st_err_norm
        model.covariance          = cov

    except (ValueError, ZeroDivisionError,np.linalg.linalg.LinAlgError):
        model.params['StdErr']    = np.nan
        model.params['StdErr[%]'] = np.nan
        model.covariance          = np.nan
        NMSE                      = np.nan
        J                         = np.nan
        K                         = np.nan
        P                         = np.nan

    res = ResultsNonLin(NMSE, model.covariance, J, model.params, K-P)
    return res


class ResultsNonLin:
    '''
    ResultsNonLin is a class to store the results of the nonlinear regression
    Objects:
        - NMSE (real): normalized mean square error
        - cov (numpy array): covariance matrix
        - jac (numpy array): jacobian matrix
        - params (pandas dataframe): estimated parameters
        - dof (int): degrees of freedom
    '''
    def __init__(self,
                 NMSE,
                 cov,
                 jac,
                 params,
                 dof):
        '''
        Constructor
        '''
        self.NMSE   = NMSE
        self.cov    = cov
        self.jac    = jac
        self.params = params
        self.dof     = dof