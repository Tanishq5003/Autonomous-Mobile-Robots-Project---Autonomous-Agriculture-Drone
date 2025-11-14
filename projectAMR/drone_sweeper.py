import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import random

# Import local classes
from DSLPIDControl import DSLPIDControl
from BaseControl import DroneModel

# --- Simulation Constants ---
SIM_HZ = 240.
CTRL_HZ = 48.
CTRL_TIMESTEP = 1.0 / CTRL_HZ
SIM_STEPS = int(SIM_HZ / CTRL_HZ)

# --- Simulation Setup ---
def setup_simulation(drone_urdf_path):
    print("Connecting to PyBullet...")
    if p.isConnected():
        p.disconnect()
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=1.0/SIM_HZ)
    planeId = p.loadURDF("plane.urdf")
    
    print(f"Loading drone from {drone_urdf_path}...")
    start_pos = [0, 0, 0.1]
    start_orn = p.getQuaternionFromEuler([0, 0, 0])
    droneId = p.loadURDF(drone_urdf_path, start_pos, start_orn)
    
    return droneId, planeId

def draw_field_boundaries(min_corner, max_corner, z=0.01):
    x_min, y_min = min_corner
    x_max, y_max = max_corner
    c0, c1 = [x_min, y_min, z], [x_max, y_min, z]
    c2, c3 = [x_max, y_max, z], [x_min, y_max, z]
    color = [0, 1, 0]
    p.addUserDebugLine(c0, c1, color, lineWidth=3)
    p.addUserDebugLine(c1, c2, color, lineWidth=3)
    p.addUserDebugLine(c2, c3, color, lineWidth=3)
    p.addUserDebugLine(c3, c0, color, lineWidth=3)

def draw_path(waypoints):
    for i in range(len(waypoints) - 1):
        p.addUserDebugLine(waypoints[i], waypoints[i+1], [1, 0, 0], lineWidth=2)

def setup_field(field_min_corner, field_max_corner, num_bad_crops=10):
    print(f"Planting {num_bad_crops} bad crops...")
    for _ in range(num_bad_crops):
        x = random.uniform(field_min_corner[0], field_max_corner[0])
        y = random.uniform(field_min_corner[1], field_max_corner[1])
        pos = [x, y, 0.05]
        vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(basePosition=pos, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

# --- Navigator Logic ---
def get_sweep_waypoints(field_min, field_max, z_hover, sweep_step=0.5):
    waypoints = []
    # 1. Takeoff (Straight up)
    waypoints.append(np.array([0, 0, z_hover]))
    
    # 2. Move to Start Corner
    waypoints.append(np.array([field_min[0], field_min[1], z_hover]))

    x_min, y_min = field_min
    x_max, y_max = field_max
    
    current_y = y_min
    going_right = True
    
    while current_y <= y_max:
        x_start = x_min if going_right else x_max
        x_end   = x_max if going_right else x_min
        
        # Row endpoints
        waypoints.append(np.array([x_start, current_y, z_hover]))
        waypoints.append(np.array([x_end,   current_y, z_hover]))
        
        # Move to next row
        current_y += sweep_step
        if current_y <= y_max + sweep_step:
            waypoints.append(np.array([x_end, current_y, z_hover]))
            
        going_right = not going_right

    # 3. Return Home and LAND
    waypoints.append(np.array([0, 0, z_hover])) # Return to home (High)
    waypoints.append(np.array([0, 0, 0.05]))    # LAND (Low)
    
    return waypoints

# --- Main Simulation Loop ---
def run_simulation(droneId, field_min_corner, field_max_corner):
    
    # --- Parameters ---
    HOVER_ALTITUDE = 1.0
    SWEEP_STEP = 0.75
    CRUISE_SPEED = 0.5     # m/s
    WAIT_TIME_AT_CORNER = 1.5 # Pause at corners to stabilize
    
    waypoints = get_sweep_waypoints(field_min_corner, field_max_corner, HOVER_ALTITUDE, SWEEP_STEP)
    draw_path(waypoints)

    ctrl = DSLPIDControl(drone_model=DroneModel.CF2P)
    
    # --- Trajectory State ---
    current_wp_idx = 0
    virtual_pos = np.array([0.0, 0.0, 0.1]) # Start on ground
    target_vel = np.zeros(3)
    
    wait_timer = 0.0 
    is_waiting = False
    
    print("Taking off...")

    while True:
        # 1. Get Drone State (FIXED: Variable names match computeControl now)
        cur_pos, cur_quat = p.getBasePositionAndOrientation(droneId)
        cur_vel, cur_ang_vel = p.getBaseVelocity(droneId)
        
        # 2. Update Virtual Target (The "Carrot")
        if current_wp_idx < len(waypoints):
            goal = waypoints[current_wp_idx]
            vector_to_goal = goal - virtual_pos
            dist_to_goal = np.linalg.norm(vector_to_goal)
            
            # Logic: Move carrot -> Reached Goal? -> Wait -> Next Goal
            if is_waiting:
                # If we are waiting, stop the carrot
                target_vel = np.zeros(3)
                wait_timer += CTRL_TIMESTEP
                
                # Drone must be close to carrot to proceed
                drone_dist = np.linalg.norm(np.array(cur_pos) - virtual_pos)
                
                if wait_timer >= WAIT_TIME_AT_CORNER and drone_dist < 0.2:
                    print(f"  -> Moving to next waypoint {current_wp_idx + 1}")
                    is_waiting = False
                    current_wp_idx += 1
                    wait_timer = 0.0
            
            elif dist_to_goal < 0.05:
                # Carrot reached the waypoint. Start waiting.
                print(f"Virtual Target arrived at {current_wp_idx}")
                is_waiting = True
                target_vel = np.zeros(3)
                virtual_pos = goal 
                
            else:
                # Move carrot towards goal
                direction = vector_to_goal / dist_to_goal
                target_vel = direction * CRUISE_SPEED
                virtual_pos += target_vel * CTRL_TIMESTEP

        else:
            # --- MISSION COMPLETE LOGIC ---
            print("Landing successful. Mission Complete.")
            # Let physics run for a moment to settle on ground, then exit
            for _ in range(100):
                p.stepSimulation()
                time.sleep(1./SIM_HZ)
            break 

        # 3. Controller
        rpm, _, _ = ctrl.computeControl(
            control_timestep=CTRL_TIMESTEP,
            cur_pos=np.array(cur_pos),
            cur_quat=np.array(cur_quat),
            cur_vel=np.array(cur_vel),
            cur_ang_vel=np.array(cur_ang_vel),
            target_pos=virtual_pos,
            target_vel=target_vel
        )
        
        # 4. Physics
        forces = ctrl.KF * (rpm**2)
        for _ in range(SIM_STEPS):
            p.applyExternalForce(droneId, 0, [0, 0, forces[0]], [0, 0, 0], p.LINK_FRAME)
            p.applyExternalForce(droneId, 1, [0, 0, forces[1]], [0, 0, 0], p.LINK_FRAME)
            p.applyExternalForce(droneId, 2, [0, 0, forces[2]], [0, 0, 0], p.LINK_FRAME)
            p.applyExternalForce(droneId, 3, [0, 0, forces[3]], [0, 0, 0], p.LINK_FRAME)
            p.stepSimulation()
            time.sleep(1./SIM_HZ)

if __name__ == "__main__":
    MIN_CORNER = [-2, -2]
    MAX_CORNER = [2, 2]
    drone_id, _ = setup_simulation("cf2p.urdf")
    draw_field_boundaries(MIN_CORNER, MAX_CORNER)
    run_simulation(drone_id, MIN_CORNER, MAX_CORNER)