import numpy as np
from numpy.linalg import inv
from control.matlab import *
import matplotlib.pyplot as plt
from control.matlab import ss, c2d

# Input : Original Continuous-time Model
# Output : Discrete-time Model
def gen_discrete_model_from_cont(Ac, Bc, Cc, Dc, Ts):
    sysd = c2d(ss(Ac,Bc,Cc,Dc),Ts)
    return sysd.A, sysd.B, sysd.C, sysd.D

# Input : Original State-space Model
# Output : Augmented State-space Model
def gen_augmented_model(Am, Bm, Cm):
    m, n = Cm.shape

    zero_n = np.zeros((m,n))
    ones_m = np.ones((m,m))

    A = np.block([[Am, zero_n.T], 
                  [Cm@Am, ones_m]])

    B = np.block([[Bm], 
                  [Cm@Bm]])

    C = np.block([zero_n, ones_m])
    
    return A, B, C

# Input : Augmented State-space Model
# Output : F, Phi, Rs of Equation {Y = F*x(k_i) + Phi*Delta_U, Rs = [1 1 1 ... 1]^T * r(k_i)} 
def gen_predicted_output(A, B, C, Np, Nc):
    m, n = C.shape
    
    F = C@A
    for i in range(Np-1):
        F = np.vstack([F, F[-1,:]@A])
    
    zero_Nc_m_1 = np.zeros((1,Nc-1))
    Phi = np.hstack([C@B, zero_Nc_m_1])
    for i in range(Np-1):
        Phi = np.vstack([Phi, np.block([F[i,:]@B, Phi[i,:-1]])])
        
    Rs = np.ones((Np,1))
    return F, Phi, Rs

# Input : F, Phi, R  (R is weighting)
# Output : (Phi^T*Phi)^-1 * Phi^T
def gen_optimal_input(F, Phi, R):
    inv_PhiT_Phi = inv(Phi.T@Phi + R)
    
    Delta_U_common_term = inv_PhiT_Phi@Phi.T
    
    return Delta_U_common_term

# Input : F, Phi, R  (R is weighting)
# Output : K_y, K_mpc
def gen_MPC_gain(F, Phi, R, Rs):
    inv_PhiT_Phi = inv(Phi.T@Phi + R)
    Delta_U_common_term = inv_PhiT_Phi@Phi.T
    K_y_term = Delta_U_common_term@Rs
    K_mpc_term = Delta_U_common_term@F
    
    return np.array(K_y_term[0]), np.array([K_mpc_term[0]])

# 
def gen_StateObserver_model(A,B,C,Kob,Ky,Kmpc):
    m, n = C.shape
    A_so = np.block([[A-Kob@C, np.zeros([n,n])],
                     [-B@Kmpc, A-B@Kmpc]])
    B_so = np.block([[np.zeros([n,m])],
                     [B*Ky]])
    
    return A_so, B_so

#####    
def sim_trial(Tstop,Ts,Ad,Bd,Cd,F,Phi,R,Rs,Kob):
    A,B,C = gen_augmented_model(Ad, Bd, Cd)
    Ky, Kmpc = gen_MPC_gain(F, Phi, R, Rs)
    
    # initial setting
    N = int(Tstop/Ts)
    t = np.arange(0, Tstop, Ts)

    u_k = 0

    # Initial condition of discrete-time model
    xm_k = np.array([[0],
                     [0]])
    # Initial condition of state-estimator
    xh_k = np.array([[0], 
                     [0], 
                     [0]])
    y_k = np.array([[0]])

    xm_kp1 = xm_k
    xh_kp1 = xh_k

    xm_data = []
    xh_data = []
    y_data = []
    u_data = []
    du_data = []
    
    # set point
    r = 1

    # simulation
    for k in range(N):   
        # Predictive Controller with State estimates
        Delta_U = Ky*r - Kmpc@xh_k
        Delta_u_k = Delta_U.item(0)
        u_k = u_k + Delta_u_k

        # Simulation of Model Update: Discrete-time state-space model
        xm_kp1 = Ad@xm_k + Bd*u_k
        # Output
        y_k = Cd@xm_k

        #  Closed loop Observer : Augmented model
        xh_kp1 = A@xh_k + B*Delta_u_k + Kob@(y_k - C@xh_k)

        # store Discrete-time model
        xm_data.append([xm_k.item(0), xm_k.item(1)])
        xh_data.append([xh_k.item(0), xh_k.item(1), xh_k.item(2)])
        y_data.append([y_k.item(0)])
        u_data.append(u_k)
        du_data.append(Delta_u_k)

        # update
        xm_k = xm_kp1
        xh_k = xh_kp1

    return t, xm_data, xh_data, y_data, u_data, du_data

def sim_trial_const_du(Tstop,Ts,Ad,Bd,Cd,F,Phi,R,Rs,Kob,M_du,Gamma_du,E_du):
    A,B,C = gen_augmented_model(Ad, Bd, Cd)
    
    # initial setting
    N = int(Tstop/Ts)
    t = np.arange(0, Tstop, Ts)

    u_k = 0

    # Initial condition of discrete-time model
    xm_k = np.array([[0],
                     [0]])
    # Initial condition of state-estimator
    xh_k = np.array([[0], 
                     [0], 
                     [0]])
    y_k = np.array([[0]])

    xm_kp1 = xm_k
    xh_kp1 = xh_k

    xm_data = []
    xh_data = []
    y_data = []
    u_data = []
    du_data = []

    # set point
    r = 1

    [n1, m1] = M_du.shape
    # simulation
    for k in range(N):   
        # Predictive Controller with State estimates
        #Delta_U = Ky*r - Kmpc@xh_k
        Delta_U = inv(Phi.T@Phi + R)@Phi.T@(Rs - F@xh_k)
        #print(k, 'ori', Delta_U)
        
        m=0
        for i in range(n1):
            if M_du[i,:]@Delta_U > Gamma_du[i]:
                m = m + 1
        if m > 0:
            F_du = -2*Phi.T@(Rs - F@xh_k)
            Delta_U, _ = QP_hild(E_du, F_du, M_du, Gamma_du)
            #print(k, 'hild', Delta_U)
      
        Delta_u_k = Delta_U.item(0) # receding horizon
        u_k = u_k + Delta_u_k

        # Simulation of Model Update: Discrete-time state-space model
        xm_kp1 = Ad@xm_k + Bd*u_k
        # Output
        y_k = Cd@xm_k

        #  Closed loop Observer : Augmented model
        xh_kp1 = A@xh_k + B*Delta_u_k + Kob@(y_k - C@xh_k)

        # store Discrete-time model
        xm_data.append([xm_k.item(0), xm_k.item(1)])
        xh_data.append([xh_k.item(0), xh_k.item(1), xh_k.item(2)])
        y_data.append([y_k.item(0)])
        u_data.append(u_k)
        du_data.append(Delta_u_k)
        
        # update
        xm_k = xm_kp1
        xh_k = xh_kp1

    return t, xm_data, xh_data, y_data, u_data, du_data

def sim_trial_const_u(Tstop,Ts,Ad,Bd,Cd,F,Phi,R,Rs,Kob,M_du,Gamma_du,E_du):
    A,B,C = gen_augmented_model(Ad, Bd, Cd)
    
    # initial setting
    N = int(Tstop/Ts)
    t = np.arange(0, Tstop, Ts)

    u_k = 0

    # Initial condition of discrete-time model
    xm_k = np.array([[0],
                     [0]])
    # Initial condition of state-estimator
    xh_k = np.array([[0], 
                     [0], 
                     [0]])
    y_k = np.array([[0]])

    xm_kp1 = xm_k
    xh_kp1 = xh_k

    xm_data = []
    xh_data = []
    y_data = []
    u_data = []
    du_data = []

    # set point
    r = 1

    [n1, m1] = M_du.shape
    # simulation
    for k in range(N):   
        # Predictive Controller with State estimates
        #Delta_U = Ky*r - Kmpc@xh_k
        Delta_U = inv(Phi.T@Phi + R)@Phi.T@(Rs - F@xh_k)
        #print(k, 'ori', Delta_U)
        
        m=0
        for i in range(n1):
            if M_du[i,:]@Delta_U > Gamma_du[i] - M_du[i,0]*u_k:
                m = m + 1
        if m > 0:
            F_du = -2*Phi.T@(Rs - F@xh_k)
            Delta_U, _ = QP_hild(E_du, F_du, M_du, Gamma_du)
            #print(k, 'hild', Delta_U)
      
        Delta_u_k = Delta_U.item(0)
        u_k = u_k + Delta_u_k

        # Simulation of Model Update: Discrete-time state-space model
        xm_kp1 = Ad@xm_k + Bd*u_k
        # Output
        y_k = Cd@xm_k

        #  Closed loop Observer : Augmented model
        xh_kp1 = A@xh_k + B*Delta_u_k + Kob@(y_k - C@xh_k)

        # store Discrete-time model
        xm_data.append([xm_k.item(0), xm_k.item(1)])
        xh_data.append([xh_k.item(0), xh_k.item(1), xh_k.item(2)])
        y_data.append([y_k.item(0)])
        u_data.append(u_k)
        du_data.append(Delta_u_k)
        
        # update
        xm_k = xm_kp1
        xh_k = xh_kp1

    return t, xm_data, xh_data, y_data, u_data, du_data

# Hildreth's Quadratic Programming
def QP_hild(E, F, M, Gamma):
    [n1, m1] = M.shape
    x_o = -inv(E)@F
    k=0
    for i in range(n1):
        if M[i,:]@x_o > Gamma[i]:
            k = k + 1
    if k == 0:
        return x_o, k
    
    # building dual problem
    P = M@inv(E)@M.T
    K = M@inv(E)@F + Gamma
    [n, m] = K.shape
    x_init = np.zeros((n,m))
    lamda = x_init
    al = 10
    k_max = 100
    k = 1
    for km in range(k_max):
        lamda_p = lamda
        for i in range(n):
            w = P[i,:]@lamda - P[i,i]*lamda[i]
            w = w + K[i]
            la = -w/P[i,i]
            lamda[i] = max(0,la)           
        conv = (lamda - lamda_p).T@(lamda - lamda_p)
        if conv < 10e-10:
            break
        k = k + 1
        
    x_star = x_o - inv(E)@M.T@lamda
    return x_star, k