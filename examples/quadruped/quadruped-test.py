# Description: This script is used to compare the nominal and lyapunov controllers


import time
import numpy as np
from tqdm import tqdm
# Gym and Simulation related imports
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr
# Control imports
from quadruped_pympc import config as cfg
from quadruped_pympc.helpers.foothold_reference_generator import FootholdReferenceGenerator
from quadruped_pympc.helpers.periodic_gait_generator import PeriodicGaitGenerator
from quadruped_pympc.helpers.swing_trajectory_controller import SwingTrajectoryController
from quadruped_pympc.helpers.terrain_estimator import TerrainEstimator
from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco
# HeightMap import
if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
    from gym_quadruped.sensors.heightmap import HeightMap
    from quadruped_pympc.helpers.visual_foothold_adaptation import VisualFootholdAdaptation
from gym_quadruped.utils.mujoco.visual import render_sphere




def main(controller_type="nominal", disturbances=np.array([0, 0, -200, 0, 0, 0])):
    np.set_printoptions(precision=3, suppress=True)

    robot_name = cfg.robot
    hip_height = cfg.hip_height
    robot_leg_joints = cfg.robot_leg_joints
    robot_feet_geom_names = cfg.robot_feet_geom_names
    scene_name = cfg.simulation_params['scene']
    simulation_dt = cfg.simulation_params['dt']

    state_observables_names = ('base_pos', 'base_lin_vel', 'base_ori_euler_xyz', 'base_ori_quat_wxyz', 'base_ang_vel',
                               'qpos_js', 'qvel_js', 'tau_ctrl_setpoint',
                               'feet_pos_base', 'feet_vel_base', 'contact_state', 'contact_forces_base',)


    # Create the quadruped robot environment -----------------------------------------------------------
    env = QuadrupedEnv(robot=robot_name,
                       hip_height=hip_height,
                       legs_joint_names=robot_leg_joints,  # Joint names of the legs DoF
                       feet_geom_name=robot_feet_geom_names,  # Geom/Frame id of feet
                       scene=scene_name,
                       sim_dt=simulation_dt,
                       ref_base_lin_vel=0.3,  # pass a float for a fixed value
                       ground_friction_coeff=1.5,  # pass a float for a fixed value
                       base_vel_command_type="forward",  # "forward", "random", "forward+rotate", "human"
                       state_obs_names=state_observables_names,  # Desired quantities in the 'state' vec
                       )


    # Some robots require a change in the zero joint-space configuration. If provided apply it
    if cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], cfg.qpos0_js))

    env.reset(random=False)
    env.render()  # Pass in the first render call any mujoco.viewer.KeyCallbackType


    # Controller initialization -------------------------------------------------------------
    mpc_frequency = cfg.simulation_params['mpc_frequency']
    mpc_dt = cfg.mpc_params['dt']
    horizon = cfg.mpc_params['horizon']

    # input_rates optimize the delta_GRF (smoooth!)
    # nominal optimize directly the GRF (not smooth)
    # sampling use GPU
    if controller_type == 'nominal':
        from quadruped_pympc.controllers.gradient.nominal.centroidal_nmpc_nominal import Acados_NMPC_Nominal

        controller = Acados_NMPC_Nominal()

    elif controller_type == 'lyapunov':
        from quadruped_pympc.controllers.gradient.lyapunov.centroidal_nmpc_lyapunov import Acados_NMPC_Lyapunov

        controller = Acados_NMPC_Lyapunov()


    # Periodic gait generator --------------------------------------------------------------
    gait_name = cfg.simulation_params['gait']
    gait_params = cfg.simulation_params['gait_params'][gait_name]
    gait_type, duty_factor, step_frequency = gait_params['type'], gait_params['duty_factor'], gait_params['step_freq']
    # Given the possibility to use nonuniform discretization,
    # we generate a contact sequence two times longer and with a dt half of the one of the mpc
    pgg = PeriodicGaitGenerator(duty_factor=duty_factor,
                                step_freq=step_frequency,
                                gait_type=gait_type,
                                horizon=horizon)
    # in the case of nonuniform discretization, we create a list of dts and horizons for each nonuniform discretization
    if (cfg.mpc_params['use_nonuniform_discretization']):
        contact_sequence_dts = [cfg.mpc_params['dt_fine_grained'], mpc_dt]
        contact_sequence_lenghts = [cfg.mpc_params['horizon_fine_grained'], horizon]
    else:
        contact_sequence_dts = [mpc_dt]
        contact_sequence_lenghts = [horizon]
    contact_sequence = pgg.compute_contact_sequence(contact_sequence_dts=contact_sequence_dts,
                                                    contact_sequence_lenghts=contact_sequence_lenghts)
    nominal_sample_freq = pgg.step_freq


    # Create the foothold reference generator ------------------------------------------------
    stance_time = (1 / pgg.step_freq) * pgg.duty_factor
    frg = FootholdReferenceGenerator(stance_time=stance_time, hip_height=cfg.hip_height, lift_off_positions=env.feet_pos(frame='world'))


    # Create swing trajectory generator ------------------------------------------------------
    step_height = cfg.simulation_params['step_height']
    swing_period = (1 - pgg.duty_factor) * (1 / pgg.step_freq)
    position_gain_fb = cfg.simulation_params['swing_position_gain_fb']
    velocity_gain_fb = cfg.simulation_params['swing_velocity_gain_fb']
    swing_generator = cfg.simulation_params['swing_generator']
    stc = SwingTrajectoryController(step_height=step_height, swing_period=swing_period,
                                    position_gain_fb=position_gain_fb, velocity_gain_fb=velocity_gain_fb,
                                    generator=swing_generator)


    # Terrain estimator -----------------------------------------------------------------------
    terrain_computation = TerrainEstimator()


    # Initialization of variables used in the main control loop --------------------------------
    # Set the reference for the state
    ref_pose = np.array([0, 0, cfg.hip_height])
    ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()
    ref_orientation = np.array([0.0, 0.0, 0.0])


    ref_state = {}

    # Starting contact sequence
    previous_contact = np.array([1, 1, 1, 1])
    previous_contact_mpc = np.array([1, 1, 1, 1])
    current_contact = np.array([1, 1, 1, 1])

    nmpc_GRFs = np.zeros((12,))
    nmpc_wrenches = np.zeros((6,))
    nmpc_footholds = np.zeros((12,))

    # Jacobian matrices
    jac_feet_prev = LegsAttr(*[np.zeros((3, env.mjModel.nv)) for _ in range(4)])
    jac_feet_dot = LegsAttr(*[np.zeros((3, env.mjModel.nv)) for _ in range(4)])
    # Torque vector
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    # State
    state_current, state_prev = {}, {}
    feet_pos = None
    feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ["FL", "FR", "RL", "RR"]


    # Create HeightMap -----------------------------------------------------------------------
    if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
        resolution_vfa = 0.04
        dimension_vfa = 7
        heightmaps = LegsAttr(FL=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mjModel=env.mjModel, mjData=env.mjData),
                        FR=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mjModel=env.mjModel, mjData=env.mjData),
                        RL=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mjModel=env.mjModel, mjData=env.mjData),
                        RR=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mjModel=env.mjModel, mjData=env.mjData))
        vfa = VisualFootholdAdaptation(legs_order=legs_order, adaptation_strategy=cfg.simulation_params['visual_foothold_adaptation'])



    # --------------------------------------------------------------
    RENDER_FREQ = 30  # Hz
    N_EPISODES = 1
    N_STEPS_PER_EPISODE = 10000
    last_render_time = time.time()

    state_obs_history, ctrl_state_history = [], []
    for episode_num in tqdm(range(N_EPISODES), desc="Episodes"):

        ep_state_obs_history, ep_ctrl_state_history = [], []
        for _ in range(N_STEPS_PER_EPISODE):
            step_start = time.time()


            # Update the robot state --------------------------------
            feet_pos = env.feet_pos(frame='world')
            hip_pos = env.hip_positions(frame='world')
            base_lin_vel = env.base_lin_vel(frame='world')
            base_ang_vel = env.base_ang_vel(frame='world')

            state_current = dict(
                position=env.base_pos,
                linear_velocity=base_lin_vel,
                orientation=env.base_ori_euler_xyz,
                angular_velocity=base_ang_vel,
                foot_FL=feet_pos.FL,
                foot_FR=feet_pos.FR,
                foot_RL=feet_pos.RL,
                foot_RR=feet_pos.RR
                )


            # Update the desired contact sequence ---------------------------
            pgg.run(simulation_dt, pgg.step_freq)
            contact_sequence = pgg.compute_contact_sequence(contact_sequence_dts=contact_sequence_dts,
                                                    contact_sequence_lenghts=contact_sequence_lenghts)


            previous_contact = current_contact
            current_contact = np.array([contact_sequence[0][0],
                                        contact_sequence[1][0],
                                        contact_sequence[2][0],
                                        contact_sequence[3][0]])


            # Compute the reference for the footholds ---------------------------------------------------
            frg.update_lift_off_positions(previous_contact, current_contact, feet_pos, legs_order)
            ref_feet_pos = frg.compute_footholds_reference(
                com_position=env.base_pos,
                base_ori_euler_xyz=env.base_ori_euler_xyz,
                base_xy_lin_vel=base_lin_vel[0:2],
                ref_base_xy_lin_vel=ref_base_lin_vel[0:2],
                hips_position=hip_pos,
                com_height_nominal=cfg.simulation_params['ref_z'])


            # Adjust the footholds given the terrain -----------------------------------------------------
            if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):

                time_adaptation = time.time()
                if(stc.check_apex_condition(current_contact, interval=0.01) and vfa.initialized == False):
                    for leg_id, leg_name in enumerate(legs_order):
                        heightmaps[leg_name].update_height_map(ref_feet_pos[leg_name], yaw=env.base_ori_euler_xyz[2])
                    vfa.compute_adaptation(legs_order, ref_feet_pos, hip_pos, heightmaps, base_lin_vel, env.base_ori_euler_xyz, base_ang_vel)
                    #print("Adaptation time: ", time.time() - time_adaptation)

                if(stc.check_full_stance_condition(current_contact)):
                    vfa.reset()

                ref_feet_pos = vfa.get_footholds_adapted(ref_feet_pos)



            # Estimate the terrain slope and elevation -------------------------------------------------------
            terrain_roll, \
                terrain_pitch, \
                terrain_height = terrain_computation.compute_terrain_estimation(
                base_position=env.base_pos,
                yaw=env.base_ori_euler_xyz[2],
                feet_pos=frg.lift_off_positions,
                current_contact=current_contact)

            ref_pos = np.array([0, 0, cfg.hip_height])
            ref_pos[2] = cfg.simulation_params['ref_z'] + terrain_height


            # Update state reference ------------------------------------------------------------------------
            ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()


            ref_state |= dict(ref_foot_FL=ref_feet_pos.FL.reshape((1, 3)),
                              ref_foot_FR=ref_feet_pos.FR.reshape((1, 3)),
                              ref_foot_RL=ref_feet_pos.RL.reshape((1, 3)),
                              ref_foot_RR=ref_feet_pos.RR.reshape((1, 3)),
                              # Also update the reference base linear velocity and
                              ref_linear_velocity=ref_base_lin_vel,
                              ref_angular_velocity=ref_base_ang_vel,
                              ref_orientation=np.array([terrain_roll, terrain_pitch, 0.0]),
                              ref_position=ref_pos
                              )
            # -------------------------------------------------------------------------------------------------



            disturbance_wrench_bound = disturbances
            env.mjData.qfrc_applied[0] = disturbance_wrench_bound[0]
            env.mjData.qfrc_applied[1] = disturbance_wrench_bound[1]
            env.mjData.qfrc_applied[2] = disturbance_wrench_bound[2]
            env.mjData.qfrc_applied[3] = disturbance_wrench_bound[3]
            env.mjData.qfrc_applied[4] = disturbance_wrench_bound[4]
            env.mjData.qfrc_applied[5] = disturbance_wrench_bound[5]


            # TODO: this should be hidden inside the controller forward/get_action method
            # Solve OCP ---------------------------------------------------------------------------------------
            if env.step_num % round(1 / (mpc_frequency * simulation_dt)) == 0:

                time_start = time.time()

                # We can recompute the inertia of the single rigid body model
                # or use the fixed one in cfg.py
                if(cfg.simulation_params['use_inertia_recomputation']):
                    inertia = env.get_base_inertia().flatten()  # Reflected inertia of base at qpos, in world frame
                else:
                    inertia = cfg.inertia.flatten()




                nmpc_GRFs, \
                nmpc_footholds, _, \
                status = controller.compute_control(state_current,
                                                    ref_state,
                                                    contact_sequence,
                                                    inertia=inertia)


                nmpc_footholds = LegsAttr(FL=nmpc_footholds[0],
                                            FR=nmpc_footholds[1],
                                            RL=nmpc_footholds[2],
                                            RR=nmpc_footholds[3])




                # If the controller is using RTI, we need to linearize the mpc after its computation
                # this helps to minize the delay between new state->control, but only in a real case.
                # Here we are in simulation and does not make any difference for now
                if (controller.use_RTI):
                    # preparation phase
                    controller.acados_ocp_solver.options_set('rti_phase', 1)
                    status = controller.acados_ocp_solver.solve()
                    # print("preparation phase time: ", controller.acados_ocp_solver.get_stats('time_tot'))




                # TODO: Indexing should not be hardcoded. Env should provide indexing of leg actuator dimensions.
                nmpc_GRFs = LegsAttr(FL=nmpc_GRFs[0:3] * current_contact[0],
                                     FR=nmpc_GRFs[3:6] * current_contact[1],
                                     RL=nmpc_GRFs[6:9] * current_contact[2],
                                     RR=nmpc_GRFs[9:12] * current_contact[3])


            # Compute Stance Torque ---------------------------------------------------------------------------
            feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
            # Compute feet velocities
            feet_vel = LegsAttr(**{leg_name: feet_jac[leg_name] @ env.mjData.qvel for leg_name in legs_order})
            # Compute jacobian derivatives of the contact points
            jac_feet_dot = (feet_jac - jac_feet_prev) / simulation_dt  # Finite difference approximation
            jac_feet_prev = feet_jac  # Update previous Jacobians
            # Compute the torque with the contact jacobian (-J.T @ f)   J: R^nv -> R^3,   f: R^3
            tau.FL = -np.matmul(feet_jac.FL[:, env.legs_qvel_idx.FL].T, nmpc_GRFs.FL)
            tau.FR = -np.matmul(feet_jac.FR[:, env.legs_qvel_idx.FR].T, nmpc_GRFs.FR)
            tau.RL = -np.matmul(feet_jac.RL[:, env.legs_qvel_idx.RL].T, nmpc_GRFs.RL)
            tau.RR = -np.matmul(feet_jac.RR[:, env.legs_qvel_idx.RR].T, nmpc_GRFs.RR)


            # Compute Swing Torque ------------------------------------------------------------------------------
            # TODO: Move contact sequence to labels FL, FR, RL, RR instead of a fixed indexing.
            # The swing controller is in the end-effector space. For its computation,
            # we save for simplicity joints position and velocities
            qpos, qvel = env.mjData.qpos, env.mjData.qvel
            # centrifugal, coriolis, gravity
            legs_mass_matrix = env.legs_mass_matrix
            legs_qfrc_bias = env.legs_qfrc_bias

            stc.update_swing_time(current_contact, legs_order, simulation_dt)

            for leg_id, leg_name in enumerate(legs_order):
                if current_contact[leg_id] == 0:  # If in swing phase, compute the swing trajectory tracking control.
                    tau[leg_name], _, _ = stc.compute_swing_control(
                        leg_id=leg_id,
                        q_dot=qvel[env.legs_qvel_idx[leg_name]],
                        J=feet_jac[leg_name][:, env.legs_qvel_idx[leg_name]],
                        J_dot=jac_feet_dot[leg_name][:, env.legs_qvel_idx[leg_name]],
                        lift_off=frg.lift_off_positions[leg_name],
                        touch_down=nmpc_footholds[leg_name],
                        foot_pos=feet_pos[leg_name],
                        foot_vel=feet_vel[leg_name],
                        h=legs_qfrc_bias[leg_name],
                        mass_matrix=legs_mass_matrix[leg_name]
                        )


            # Set control and mujoco step ----------------------------------------------------------------------
            action = np.zeros(env.mjModel.nu)
            action[env.legs_tau_idx.FL] = tau.FL
            action[env.legs_tau_idx.FR] = tau.FR
            action[env.legs_tau_idx.RL] = tau.RL
            action[env.legs_tau_idx.RR] = tau.RR

            action_noise = np.random.normal(0, 2, size=env.mjModel.nu)
            action += action_noise

            state, reward, is_terminated, is_truncated, info = env.step(action=action)


            # Store the history of observations and control -------------------------------------------------------
            ep_state_obs_history.append(state)
            base_lin_vel_err = ref_base_lin_vel - base_lin_vel
            base_ang_vel_err = ref_base_ang_vel - base_ang_vel
            base_poz_z_err = ref_pos[2] - env.base_pos[2]
            ctrl_state = np.concatenate((base_lin_vel_err, base_ang_vel_err, [base_poz_z_err], pgg._phase_signal))
            ep_ctrl_state_history.append(ctrl_state)


            # Render only at a certain frequency -----------------------------------------------------------------
            if time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1:
                _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)

                # Plot the swing trajectory
                feet_traj_geom_ids = plot_swing_mujoco(viewer=env.viewer,
                                                       swing_traj_controller=stc,
                                                       swing_period=stc.swing_period,
                                                       swing_time=LegsAttr(FL=stc.swing_time[0],
                                                                           FR=stc.swing_time[1],
                                                                           RL=stc.swing_time[2],
                                                                           RR=stc.swing_time[3]),
                                                       lift_off_positions=frg.lift_off_positions,
                                                       nmpc_footholds=nmpc_footholds,
                                                       ref_feet_pos=ref_feet_pos,
                                                       geom_ids=feet_traj_geom_ids)


                # Update and Plot the heightmap
                if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
                    #if(stc.check_apex_condition(current_contact, interval=0.01)):
                    for leg_id, leg_name in enumerate(legs_order):
                        data = heightmaps[leg_name].data#.update_height_map(ref_feet_pos[leg_name], yaw=env.base_ori_euler_xyz[2])
                        if(data is not None):
                            for i in range(data.shape[0]):
                                for j in range(data.shape[1]):
                                        heightmaps[leg_name].geom_ids[i, j] = render_sphere(viewer=env.viewer,
                                                                                            position=([data[i][j][0][0],data[i][j][0][1],data[i][j][0][2]]),
                                                                                            diameter=0.01,
                                                                                            color=[0, 1, 0, .5],
                                                                                            geom_id=heightmaps[leg_name].geom_ids[i, j]
                                                                                            )

                # Plot the GRF
                for leg_id, leg_name in enumerate(legs_order):
                    feet_GRF_geom_ids[leg_name] = render_vector(env.viewer,
                                                                vector=feet_GRF[leg_name],
                                                                pos=feet_pos[leg_name],
                                                                scale=np.linalg.norm(feet_GRF[leg_name]) * 0.005,
                                                                color=np.array([0, 1, 0, .5]),
                                                                geom_id=feet_GRF_geom_ids[leg_name])

                env.render()
                last_render_time = time.time()


            # Reset the environment if the episode is terminated ------------------------------------------------
            if env.step_num > N_STEPS_PER_EPISODE or is_terminated or is_truncated:
                if is_terminated:
                    print("Environment terminated")
                else:
                    state_obs_history.append(ep_state_obs_history)
                    ctrl_state_history.append(ep_ctrl_state_history)
                env.reset(random=False)
                pgg.reset()
                if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'): vfa.reset()
                frg.lift_off_positions = env.feet_pos(frame='world')
                current_contact = np.array([0, 0, 0, 0])
                previous_contact = np.asarray(current_contact)


    env.close()






if __name__ == '__main__':
    main(controller_type="nominal", disturbances=np.array([0, 0, -200, 0, 0, 0]))
    main(controller_type="lyapunov", disturbances=np.array([0, 0, -200, 0, 0, 0]))