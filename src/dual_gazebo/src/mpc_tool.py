import numpy as np
import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
sys.path.append('../../../')

# Import do_mpc package:
import do_mpc


def set_model(init_angle):

    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)


    m0 = 11  # kg, mass of the cart
    m1 = 10  # kg, mass of the first rod
    L1 = 0.5  # m,  length of the first rod

    g = 9.80665 # m/s^2, Gravity


    l1 = L1/2 # m,

    J1 = (m1 * l1**2) / 3   # Inertia


    h1 = m0 + m1
    h2 = m1*l1

    h4 = m1*l1**2 +  J1
    h7 = (m1*l1 ) * g


    pos = model.set_variable('_x',  'pos')
    theta = model.set_variable('_x',  'theta', (1,1))
    dpos = model.set_variable('_x',  'dpos')
    dtheta = model.set_variable('_x',  'dtheta', (1,1))

    u = model.set_variable('_u',  'force')


    ddpos = model.set_variable('_z', 'ddpos')
    ddtheta = model.set_variable('_z', 'ddtheta', (1,1))


    model.set_rhs('pos', dpos)
    model.set_rhs('theta', dtheta)
    model.set_rhs('dpos', ddpos)

    euler_lagrange = vertcat(
            # 1
            h1*ddpos+h2*ddtheta[0]*cos(theta[0])
            - (h2*dtheta[0]**2*sin(theta[0]) + u),
            # 2
            h2*cos(theta[0])*ddpos + h4*ddtheta[0]
            - (h7*sin(theta[0])),
            )


    model.set_alg('euler_lagrange', euler_lagrange)
    model.set_rhs('dtheta', ddtheta)


    E_kin_cart = 1 / 2 * m0 * dpos**2
    E_kin_p1 = 1 / 2 * m1 * (
        (dpos + l1 * dtheta[0] * cos(theta[0]))**2 +
        (l1 * dtheta[0] * sin(theta[0]))**2) + 1 / 2 * J1 * dtheta[0]**2

    E_kin = E_kin_cart + E_kin_p1 

    E_pot = m1 * g * l1 * cos(
    theta[0]) 

    model.set_expression('E_kin', E_kin)
    model.set_expression('E_pot', E_pot)


    model.setup()

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 30,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 0.04,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 3,
        'collocation_ni': 1,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
    }
    mpc.set_param(**setup_mpc)

    mterm = model.aux['E_kin'] - model.aux['E_pot'] # terminal cost
    lterm = model.aux['E_kin'] - model.aux['E_pot'] # stage cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    # Input force is implicitly restricted through the objective.
    mpc.set_rterm(force=0.01)


    mpc.bounds['lower','_u','force'] = -150
    mpc.bounds['upper','_u','force'] = 150

    mpc.bounds['lower','_x','theta'] = -0.174533
    mpc.bounds['upper','_x','theta'] = 0.174533

    mpc.setup()

    estimator = do_mpc.estimator.StateFeedback(model)

    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        # Note: cvode doesn't support DAE systems.
        'integration_tool': 'idas',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 0.01
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

    simulator.setup()

    return mpc, x0, u0



    n_steps = 100

    data_list_x = []
    data_list_u = []

    for k in range(n_steps):
        u0 = mpc.make_step(x0)
        print(k)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)
        print("--------------------------------------------------")

        print(x0)
        data_list_x.append(x0)
        data_list_u.append(u0)
        print("--------------------------------------------------")

    for i in range(len(data_list_x)):
        print(i, data_list_x[i], data_list_u[i])
