#!/usr/bin/env python3
import subprocess
import time
import random
import math

def teleport_box(name, x, y, z=0.025, yaw=0.0):
    """Teleports a Gazebo model using the /world/pick_place/set_pose service."""
    import math
    # Convert yaw to quaternion
    qx, qy, qz = 0.0, 0.0, math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    req = (f'name: "{name}", '
           f'position: {{x: {x:.3f}, y: {y:.3f}, z: {z:.3f}}}, '
           f'orientation: {{x: {qx:.4f}, y: {qy:.4f}, z: {qz:.4f}, w: {qw:.4f}}}')

    subprocess.run([
        "gz", "service", "-s", "/world/pick_place/set_pose",
        "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
        "--timeout", "1000", "--req", req
    ], check=True, stdout=subprocess.DEVNULL)

def main():
    trials = 5
    print(f"Starting {trials} automated random tests...")
    print("Make sure Gazebo is running before starting this!")

    for i in range(trials):
        print(f"\n--- Trial {i+1} of {trials} ---")
        
        # 1. Reset Box A to (0.5, 0.5)
        # Box A center is at z=0.1 (since it's 20cm tall)
        teleport_box("box_a", 0.5, 0.5, z=0.1, yaw=0.0)
        
        # 2. Generate a random location for Box B
        while True:
            r = random.uniform(0.40, 0.70)
            theta = random.uniform(math.radians(-120), math.radians(120))
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            
            # Ensure Box B doesn't spawn inside Box A
            if math.hypot(x - 0.5, y - 0.5) > 0.25:
                break
                
        yaw = random.uniform(-math.pi, math.pi)
        
        print(f"Teleporting Box B to: X={x:.2f}, Y={y:.2f}, Yaw={math.degrees(yaw):.1f} deg")
        teleport_box("box_b", x, y, z=0.025, yaw=yaw)
        
        # 3. Wait for the physics engine to settle the box
        time.sleep(1.0)
        
        # 4. Run the pick and place node
        print("Executing Pick-and-Place...")
        subprocess.run(["ros2", "run", "gp7_sim", "pick_place.py"])
        
        # 5. Wait a moment before the next trial
        time.sleep(2.0)

    print("\nAll trials complete!")

if __name__ == "__main__":
    main()