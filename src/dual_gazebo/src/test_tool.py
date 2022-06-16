import numpy as np
import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
sys.path.append('../../../')

# Import do_mpc package:
import do_mpc

from dmpc import *
from control.matlab import ss, c2d, tf2ss, pade, tf, series

def set_model(init_angle):

    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # States struct (optimization variables):
    pos = model.set_variable(var_type='_x', var_name='pos', shape=(1,1))
    theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
    dpos = model.set_variable(var_type='_x', var_name='dpos', shape=(1,1))
    dtheta = model.set_variable(var_type='_x', var_name='dtheta', shape=(1,1))

    # Input struct (optimization variables):
    u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))


    M = 2  # mass of the cart 
    m = 15  # mass of the pendulum    
    b = 0.1  # coefficient of friction for cart   
    I = 0.2124  # mass moment of inertia of the pendulum 
    g = -9.8
    l = 0.3 # length to pendulum center of mass  

    p = I*(M+m)+M*m*l**2

    A = [[0, 0, 1, 0],
        [ 0,0,0,1],
        [0 ,(m**2*g*l**2)/p,  -(I+m*l**2)*b/p,   0],
        [0, m*g*l*(M+m)/p, -(m*l*b)/p,  0]]

    B =[ [0],
        [0],
        [(I+m*l**2)/p],
        [m*l/p]]
    
    C = [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]

    D = [[0],
    [0],
    [0],
    [0]]


    # C = [[1, 0, 0, 0],
    # [0, 0, 1, 0]]

    # D = [[0],
    #     [0]]


    Ad = c2d(ss(A,B,C,D),0.5).A
    Bd = c2d(ss(A,B,C,D),0.5).B
    Cd = c2d(ss(A,B,C,D),0.5).C
    Dd = c2d(ss(A,B,C,D),0.5).D

    x_1_next = Ad[:,0][0]*pos + Ad[:,1][0]*theta + Ad[:,2][0]*dpos + Ad[:,3][0]*dtheta + Bd[:,0][0]*u 
    x_2_next = Ad[:,0][1]*pos + Ad[:,1][1]*theta + Ad[:,2][1]*dpos + Ad[:,3][1]*dtheta + Bd[:,0][1]*u 
    x_3_next = Ad[:,0][2]*pos + Ad[:,1][2]*theta + Ad[:,2][2]*dpos + Ad[:,3][2]*dtheta + Bd[:,0][2]*u 
    x_4_next = Ad[:,0][3]*pos + Ad[:,1][3]*theta + Ad[:,2][3]*dpos + Ad[:,3][3]*dtheta + Bd[:,0][3]*u  

    model.set_rhs('pos', x_1_next)
    model.set_rhs('theta', x_2_next)
    model.set_rhs('dpos', x_3_next)
    model.set_rhs('dtheta', x_4_next)

    model.set_expression(expr_name='cost', expr=sum1(pos**2 + theta**2 + dpos**2))

    model.setup()

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 150,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 0.01,
        # 'state_discretization': 'collocation',
        # 'collocation_type': 'radau',
        # 'collocation_deg': 3,
        # 'collocation_ni': 1,
        'store_full_solution': False,
        'store_lagr_multiplier' : False,
        'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
        # 'nlpsol_opts': {'ipopt.linear_solver': 'ma27'}
        # Use MA27 linear solver in ipopt for faster calculations:
    }
    # setup_mpc = {
    # 'n_horizon': 2,
    # 't_step': 0.05,
    # 'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}

    # }
    mpc.set_param(**setup_mpc)

    mterm = model.aux['cost']
    lterm = model.aux['cost'] # terminal cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=0.1)

    # mpc.bounds['lower','_u','force'] = -10
    # mpc.bounds['upper','_u','force'] = 10

    # mpc.bounds['lower','_x','theta'] = -0.174533 * 1
    # mpc.bounds['upper','_x','theta'] = 0.174533 * 1

    # mpc.bounds['lower','_x','pos'] = -1
    # mpc.bounds['upper','_x','pos'] = 1

    mpc.setup()

    estimator = do_mpc.estimator.StateFeedback(model)

    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        # Note: cvode doesn't support DAE systems.
        'integration_tool': 'idas',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 0.03
    }

    simulator.set_param(**params_simulator)

    simulator.x0['theta'] = init_angle  # 0 deg

    x0 = simulator.x0.cat.full()

    mpc.x0 = x0
    estimator.x0 = x0

    mpc.set_initial_guess()


    u0 = mpc.make_step(x0)


    # Quickly reset the history of the MPC data object.
    mpc.reset_history()

    return mpc, estimator, u0


