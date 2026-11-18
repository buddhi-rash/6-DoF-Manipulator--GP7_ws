import numpy as np

def fk_tcp(q, tool_offset=0.14):
    """
    Calculates Forward Kinematics for the GP7 using the standard DH table.
    Returns a 4x4 homogeneous transformation matrix of the Tool Centre Point.
    """
    # GP7 DH Parameters (from the provided table)
    d = [0.330, 0.0, 0.0, -0.440, 0.0, -0.080]
    a = [0.040, 0.445, 0.040, 0.0, 0.0, 0.0]
    alpha = [np.radians(-90), np.radians(180), np.radians(-90), 
             np.radians(90), np.radians(-90), np.radians(180)]

    T = np.eye(4)
    for i in range(6):
        # Apply the -90 deg offset for Joint 2 as specified in the DH table
        theta = q[i] if i != 1 else q[i] - np.pi/2
        
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha[i])
        sa = np.sin(alpha[i])
        
        # Standard DH Transformation Matrix
        Ti = np.array([
            [ct, -st*ca,  st*sa, a[i]*ct],
            [st,  ct*ca, -ct*sa, a[i]*st],
            [0,   sa,     ca,    d[i]],
            [0,   0,      0,     1]
        ])
        T = T @ Ti

    # The grasp point is offset along the Z-axis of Frame 6
    T_offset = np.eye(4)
    T_offset[2, 3] = tool_offset
    
    return T @ T_offset