import os
import numpy as np
import xml.etree.ElementTree as etxml
from enums import DroneModel # Import from your enums file

class BaseControl(object):
    """Base class for control."""

    def __init__(self, drone_model: DroneModel, g: float=9.8):
        self.DRONE_MODEL = drone_model
        
        # Load parameters from URDF
        self.M = self._getURDFParameter('m')
        self.GRAVITY = g * self.M
        self.KF = self._getURDFParameter('kf')
        self.KM = self._getURDFParameter('km')
        self.I = np.diag([self._getURDFParameter('ixx'), self._getURDFParameter('iyy'), self._getURDFParameter('izz')])
        self.INV_I = np.linalg.inv(self.I)
        self.L = self._getURDFParameter('arm')
        self.reset()

    def reset(self):
        self.control_counter = 0

    def computeControlFromState(self, control_timestep, state, target_pos, target_rpy=np.zeros(3), target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        return self.computeControl(control_timestep, state[0:3], state[3:7], state[10:13], state[13:16], target_pos, target_rpy, target_vel, target_rpy_rates)

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_rpy=np.zeros(3), target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        raise NotImplementedError

    def setPIDCoefficients(self, p_coeff_pos=None, i_coeff_pos=None, d_coeff_pos=None, p_coeff_att=None, i_coeff_att=None, d_coeff_att=None):
        ATTR_LIST = ['P_COEFF_FOR', 'I_COEFF_FOR', 'D_COEFF_FOR', 'P_COEFF_TOR', 'I_COEFF_TOR', 'D_COEFF_TOR']
        if not all(hasattr(self, attr) for attr in ATTR_LIST):
            print("[ERROR] PID coefficients not found.")
            exit()
        else:
            self.P_COEFF_FOR = self.P_COEFF_FOR if p_coeff_pos is None else p_coeff_pos
            self.I_COEFF_FOR = self.I_COEFF_FOR if i_coeff_pos is None else i_coeff_pos
            self.D_COEFF_FOR = self.D_COEFF_FOR if d_coeff_pos is None else d_coeff_pos
            self.P_COEFF_TOR = self.P_COEFF_TOR if p_coeff_att is None else p_coeff_att
            self.I_COEFF_TOR = self.I_COEFF_TOR if i_coeff_att is None else i_coeff_att
            self.D_COEFF_TOR = self.D_COEFF_TOR if d_coeff_att is None else d_coeff_att

    def _getURDFParameter(self, parameter_name: str):
        # MODIFIED: Handles Enum value for filename and local path
        urdf_name = self.DRONE_MODEL.value + ".urdf" 
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), urdf_name)
        
        if not os.path.exists(path):
            print(f"[ERROR] Could not find {urdf_name} in {os.path.dirname(os.path.abspath(__file__))}")
            exit()
            
        URDF_TREE = etxml.parse(path).getroot()

        if parameter_name == 'm':
            return float(URDF_TREE[1][0][1].attrib['value'])
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(URDF_TREE[1][0][2].attrib[parameter_name])
        elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff' 'prop_radius', 'drag_coeff_xy', 'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3']:
            return float(URDF_TREE[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
            return COLLISION_SHAPE_OFFSETS[2]