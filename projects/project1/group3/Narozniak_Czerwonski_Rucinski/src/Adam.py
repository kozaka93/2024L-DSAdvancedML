import numpy as np


class AdamOptim():

    """
    The Adam optimizer.

    Parameters:
    - eta (float): The learning rate. Default is 0.01.
    - beta1 (float): The exponential decay rate for the first moment estimates. Default is 0.9.
    - beta2 (float): The exponential decay rate for the second moment estimates. Default is 0.999.
    - epsilon (float): A small constant for numerical stability. Default is 1e-8.

    Attributes:
    - m_dw (float): The first moment estimate for the weight gradients.
    - v_dw (float): The second moment estimate for the weight gradients.
    - m_db (float): The first moment estimate for the bias gradients.
    - v_db (float): The second moment estimate for the bias gradients.
    """

    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        
    def update(self, t, w, b, dw, db):
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db)

        m_dw_corr = self.m_dw/(1-self.beta1**t)
        m_db_corr = self.m_db/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        v_db_corr = self.v_db/(1-self.beta2**t)


        v_dw_corr = v_dw_corr.astype(float)
        v_db_corr = v_db_corr.astype(float)
        w = w - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        b = b - self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
        return w, b