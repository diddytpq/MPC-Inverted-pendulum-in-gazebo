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

    u = model.set_variable('_u',  'force')




    _x = model.set_variable(var_type='_x', var_name='x', shape=(4,1))
    _u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

    M = 6  # mass of the cart 
    m = 15  # mass of the pendulum    
    b = 0.1  # coefficient of friction for cart   
    I = 0.2124  # mass moment of inertia of the pendulum 
    g = 9.8
    l = 0.3 # length to pendulum center of mass  

    p = I*(M+m)+M*m*l**2

    A = [[0, 1, 0, 0],
        [0 ,-(I+m*l**2)*b/p,  (m**2*g*l**2)/p,   0],
    [ 0,0,0,1],
        [0, -(m*l*b)/p,m*g*l*(M+m)/p,  0]]
    B =[ [0],
        [(I+m*l**2)/p],
            [0],
            [m*l/p]]
    C = [[1, 0, 0, 0],
        [0 ,1, 0, 0],]
    D = [[0],
        [0]]


    Ad = c2d(ss(A,B,C,D),0.05).A
    Bd = c2d(ss(A,B,C,D),0.05).B
    Cd = c2d(ss(A,B,C,D),0.05).C
    Dd = c2d(ss(A,B,C,D),0.05).D

    x_next = Ad@_x + Bd@_u

    model.set_rhs('x', x_next)


    model.set_expression(expr_name='cost', expr=sum1(_x**2))



    model.setup()

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 10,
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

    mterm =  model.aux['cost'] # terminal cost
    lterm =  model.aux['cost'] # stage cost



    # mterm = model.aux['E_kin'] - model.aux['E_pot']
    # lterm = -model.aux['E_pot']+10*(model.x['pos'])**2 # stage cost


    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=25)

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

    # simulator.x0['theta'] = init_angle  # 0 deg

    x0 = simulator.x0.cat.full()

    mpc.x0 = x0
    estimator.x0 = x0

    mpc.set_initial_guess()


    u0 = mpc.make_step(x0)


    # Quickly reset the history of the MPC data object.
    mpc.reset_history()

    simulator.setup()

    return mpc, estimator, u0


