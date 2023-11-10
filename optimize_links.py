# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 23:15:22 2023

@author: user
"""

import numpy as np
from scipy.optimize import minimize

def points(n = 100, center = [3,3], radius = 1, dis = 20, shape = "sphere"):
    """Creates an array with points in a shape"""
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


def transformation_matrix(theta, a, d, alpha, axis1 = "x", axis2 = "y"):
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
   
def optimize_links(n, dis, center, radius, shape):
    """Optimize length of links given
        n: number of points
        shape: circle or sphere
        dis: distance to the shape
        center: 2D center of the shape
        radius: radius of the shape"""
    
    trajectory = points(n, center, radius, dis, shape)
    
    def objective_function2(lengths, *args):
        trajectory = args[0]
        error = optimize(trajectory, *lengths) + np.sum([length**2 for length in lengths])
        print(error)
        return error
    
    initial_lengths = [4.61272984*2, 1.93626943*2, 0.82897907*2, 0.82897907*2, 4.69030225*2]
    
    bounds = [(0, dis/2),
      (0, dis/2), 
      (0, dis/2),  
      (0, dis/2),
      (0, dis/2)]  

    # Use optimization to adjust links lengths to minimize the error
    result = minimize(objective_function2, initial_lengths, args=(trajectory,), bounds = bounds)
    optimized_lengths = result.x
    print(optimized_lengths)
    
    return optimized_lengths


def optimize(trajectory, L0_max, L1, L2, L3, L4):
    """Return the error between the end effector position and the desired positions
    At each point, we optimize the parameters: theta1, theta2,... 
    and compute the error between the end effector position and the desired position"""
    current_step = 0
    error = 0
    params = np.zeros((100,4))
    while current_step < len(trajectory):
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
        
        params[current_step:, :] = np.array([theta1, theta2, theta3, L0])

        # Calculate the end effector position
        end_effector_position = calculate_end_effector_position(*optimized_joint_angles)
        
        error += np.linalg.norm(end_effector_position-desired_position)
        
        current_step += 1
        
    return error
        

