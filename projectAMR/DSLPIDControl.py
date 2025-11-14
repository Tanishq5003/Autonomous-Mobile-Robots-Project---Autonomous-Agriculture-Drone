import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from BaseControl import BaseControl
from enums import DroneModel

class DSLPIDControl(BaseControl):
    def __init__(self, drone_model: DroneModel, g: float=9.8):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()

        # --- TUNING FOR STABLE TRACKING ---
        self.P_COEFF_FOR = np.array([0.4, 0.4, 1.25]) # Balanced position hold
        self.I_COEFF_FOR = np.array([0.05, 0.05, 0.05]) 
        self.D_COEFF_FOR = np.array([0.2, 0.2, 0.5]) # Damping to prevent overshoot
        
        self.P_COEFF_TOR = np.array([70000.0, 70000.0, 60000.0])
        self.I_COEFF_TOR = np.array([0.0, 0.0, 500.0])
        self.D_COEFF_TOR = np.array([20000.0, 20000.0, 12000.0])
        
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([[-.5, -.5, -1], [-.5, .5, 1], [.5, .5, -1], [.5, -.5, 1]])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([[0, -1, -1], [+1, 0, 1], [0, 1, -1], [-1, 0, 1]])
        self.reset()

    def reset(self):
        super().reset()
        self.last_rpy = np.zeros(3)
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_rpy=np.zeros(3), target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep, cur_pos, cur_quat, cur_vel, target_pos, target_rpy, target_vel)
        rpm = self._dslPIDAttitudeControl(control_timestep, thrust, cur_quat, computed_target_rpy, target_rpy_rates)
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
    
    def _dslPIDPositionControl(self, control_timestep, cur_pos, cur_quat, cur_vel, target_pos, target_rpy, target_vel):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        
        # Errors
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        
        # PID Thrust Calculation
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        
        # --- SAFETY CLAMP: Prevent flipping ---
        # Limit max tilt to 30 degrees (0.52 radians)
        MAX_TILT = 0.5
        target_euler[0] = np.clip(target_euler[0], -MAX_TILT, MAX_TILT) # Roll
        target_euler[1] = np.clip(target_euler[1], -MAX_TILT, MAX_TILT) # Pitch

        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
            
        return thrust, target_euler, pos_e

    def _dslPIDAttitudeControl(self, control_timestep, thrust, cur_quat, target_euler, target_rpy_rates):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -2., 2.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) + np.multiply(self.D_COEFF_TOR, rpy_rates_e) + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST