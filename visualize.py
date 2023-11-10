# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 23:15:22 2023

@author: user
"""


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import time

# Define the parameters of your robot
# Lengths of the three links
L0_max, L1, L2, L3, L4 = 9.22545968, 3.87253886, 1.65795814, 1.65795814, 9.3806045
current_step = 0
real_traj = np.zeros((100,3))
# Joints parameters theta1, theta2, theta3, L0_max
params = np.zeros((100,4))

#Define the trajectory/Volume
def points(n = 100, center = [3,3], radius = 1, dis = 20, shape = "sphere"):
    """Creates an array with points on a shape"""
    if shape == "circle":
        output = np.array([[center[0]+radius * np.cos(2*np.pi*t/n), -dis, center[1]+radius*np.sin(2*np.pi*t/n)] for t in range(n)])
        
    elif shape == "sphere":
        x_center, y_center, z_center = center[0], center[1], -dis
        
        # Generate 100 random points in spherical coordinates
        theta = 2 * np.pi * np.random.rand(n)  # Azimuthal angle
        phi = np.arccos(2 * np.random.rand(n) - 1)  # Polar angle
        
        # Convert spherical coordinates to Cartesian coordinates
        x = x_center + radius * np.sin(phi) * np.cos(theta)
        y = y_center + radius * np.sin(phi) * np.sin(theta)
        z = z_center + radius * np.cos(phi)
        
        output = np.array([[y[i], z[i], x[i]] for i in range(n)])
    return output

# Transformation matrices for each joint
def transformation_matrix(theta, a, d, alpha, axis1 = "x", axis2 = "y"):
    """Transformation matrix given the parameters"""
    def rotation(axis, angle):
        d = {"x" : np.array([[1,0,0,0],
                             [0,np.cos(angle), -np.sin(angle), 0],
                             [0, np.sin(angle), np.cos(angle), 0],
                             [0,0,0,1]]),
             "y" : np.array([[np.cos(angle),0,np.sin(angle),0],
                            [0,1,0, 0],
                            [-np.sin(angle), 0, np.cos(angle), 0],
                            [0,0,0,1]]),
        "z" : np.array([[np.cos(angle),-np.sin(angle),0,0],
                           [np.sin(angle), np.cos(angle),0,0],
                           [0,0,1,0],
                           [0,0,0,1]])}
        return d[axis]
    
    def translation(pos, l):
        D = np.eye(4)
        D[pos, 3] = l
        return D
    
    T = rotation(axis1, alpha) @ translation(0, a) @ rotation(axis2, theta) @ translation(2, d)
    
    return T

def update(val): 
    """Updates 3D  plot"""
    global real_traj
    global current_step
    global params
    theta1 = theta1_slider.val
    theta2 = theta2_slider.val
    theta3 = theta3_slider.val
    L0 = L0_slider.val

    # Calculate the transformation matrices for each joint
    T0 = transformation_matrix(np.deg2rad(-90), L0, 0, 0, "x", "z")
    T1 = transformation_matrix(np.deg2rad(theta1), L1, 0, np.pi/2, "z", "y")
    T2 = transformation_matrix(np.deg2rad(theta2), L2, 0, np.pi/2, "x", "z")
    T3 = transformation_matrix(np.deg2rad(theta3), L3, 0, 0, "z", "x")
    T4 = transformation_matrix(-np.pi/3, L4, 0, np.pi/2, "y", "z")

    # Base location
    base = [0, 0, 0, 1]

    # Define the initial and final points of each link
    points = np.array([base,
                       np.linalg.inv(T0[[1,0,2,3]]).dot(base),
                       np.linalg.inv(T1 @ T0[[1,0,2,3]]).dot(base),
                       np.linalg.inv(T2 @ T1 @ T0[[1,0,2,3]]).dot(base),
                       np.linalg.inv(T3 @ T2 @ T1 @ T0[[1,0,2,3]]).dot(base),
                       np.linalg.inv(T4 @ T3 @ T2 @ T1 @ T0[[1,0,2,3]]).dot(base)])

    # Extract x, y, and z coordinates for plotting
    y = points[:, 0]
    z = points[:, 1]
    x = points[:, 2]
    
    
    real_traj[current_step:, :] = np.array([x[-1], y[-1], z[-1]])
    
    # Clear the previous plot and update it
    ax.cla()
    ax.scatter(real_traj[:,0], real_traj[:,1], real_traj[:,2], color ="r")
    ax.plot(x, y, z, marker='o', linestyle='-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-22, 22)
    ax.set_ylim(-22, 22)
    ax.set_zlim(-22, 22)
    plt.draw()

def plot_params():
    """Plot the parameters after optimization"""
    global params
    # Create a 4x1 grid of subplots
    fig2, axs = plt.subplots(4, 1, figsize=(8, 10))
    
    # Plot each data array in a separate subplot
    for i in range(4):
        axs[i].plot(params[:, i])
        axs[i].set_title(f'Subplot {i+1}')
        axs[i].set_xlabel('Time')
        if i==3:
            axs[i].set_ylabel('L0')
        else:
            axs[i].set_ylabel('theta'+str(i))
    
    # Adjust spacing between subplots
    plt.tight_layout()
  

# Modify the update function to update the sliders based on the trajectory
def optimize():
    """Optimize the """
    global current_step
    global params
    
    if current_step < len(trajectory):
        x_goal, y_goal, z_goal = trajectory[current_step]
        
        # Calculate the inverse kinematics to set the values of theta1, theta2, theta3, and L0
        # Define a function to calculate the end effector's position based on joint angles
        def calculate_end_effector_position(theta1, theta2, theta3, L0):
            # Calculate transformation matrices for each joint
            T0 = transformation_matrix(np.deg2rad(-90), L0, 0, 0, "x", "z")
            T1 = transformation_matrix(np.deg2rad(theta1), L1, 0, np.pi/2, "z", "y")
            T2 = transformation_matrix(np.deg2rad(theta2), L2, 0, np.pi/2, "x", "z")
            T3 = transformation_matrix(np.deg2rad(theta3), L3, 0, 0, "z", "x")
            T4 = transformation_matrix(-np.pi/3, L4, 0, np.pi/2, "y", "z")

            # Calculate the end effector position
            end_effector_position = np.linalg.inv(T4 @ T3 @ T2 @ T1 @ T0[[1,0,2,3]]).dot([0, 0, 0, 1])
            
            return end_effector_position[:3]  # Return only the x, y, and z coordinates

        # Define an objective function to minimize the error between current and desired positions
        def objective_function(joint_angles, *args):
            desired_position = args[0]
            current_position = calculate_end_effector_position(*joint_angles)
            error = np.linalg.norm(current_position - desired_position)
            return error

        # Define the initial joint angle values
        initial_joint_angles = list(params[-1,:])

        # Define the desired end effector position from the trajectory
        desired_position = trajectory[current_step]
        
        bounds = [(0, 360),  # Bounds for theta1
          (0, 360),  # Bounds for theta2
          (0, 360),  # Bounds for theta3
          (0, L0_max)]  # Bounds for L0

        # Use optimization to adjust joint angles to minimize the error
        result = minimize(objective_function, initial_joint_angles, args=(desired_position,), bounds = bounds)
        optimized_joint_angles = result.x

        # Set the optimized joint angles as the new values for theta1, theta2, theta3, and L0
        theta1, theta2, theta3, L0 = optimized_joint_angles
        
        # Save the parameters used
        params[current_step:, :] = np.array([theta1, theta2, theta3, L0])
        
        theta1_slider.set_val(theta1)
        theta2_slider.set_val(theta2)
        theta3_slider.set_val(theta3)
        L0_slider.set_val(L0)
        
        current_step += 1
        print(current_step)
    else:
        # Trajectory has been completed
        pass

#Example of sphere
trajectory = points(n = 100, center = [3,3], radius = 1, dis = 18, shape = "sphere")

#Example of circle
trajectory = points(n = 100, center = [3,3], radius = 2, dis = 18, shape = "circle")
    
# Create the initial figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create sliders for theta1, theta2, and theta3 below the plot
theta3_slider_ax = plt.axes([0.15, 0.02, 0.65, 0.03])
theta2_slider_ax = plt.axes([0.15, 0.06, 0.65, 0.03])
theta1_slider_ax = plt.axes([0.15, 0.10, 0.65, 0.03])
L0_slider_ax = plt.axes([0.15, 0.14, 0.65, 0.03])

theta1_slider = Slider(theta1_slider_ax, 'Theta1', 0, 360, valinit=0)
theta2_slider = Slider(theta2_slider_ax, 'Theta2', 0, 360, valinit=180)
theta3_slider = Slider(theta3_slider_ax, 'Theta3', 0, 360, valinit=0)
L0_slider = Slider(L0_slider_ax, 'L0', 1, L0_max, valinit=1)

# Attach the update function to the sliders
theta1_slider.on_changed(update)
theta2_slider.on_changed(update)
theta3_slider.on_changed(update)
L0_slider.on_changed(update)

plt.show()



