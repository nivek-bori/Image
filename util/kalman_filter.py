import numpy as np

class KalmanFilter:
    def __init__(self, bbox):
        # x: [cx, cy, aspect_ratio, height, vx, vy, vh]
        
        # calculating to center coords and aspect ratio
        cx = bbox[0]
        cy = bbox[1]
        w, h = bbox[2], bbox[3]
        aspect_ratio = w / h if h != 0 else 1.0
        
        self.x = np.array([[cx], [cy], [aspect_ratio], [h], [0], [0], [0]], dtype=np.float32) # velocity is initally zero
        
        # State transition matrix (constant velocity model)
        dt = 1  # Time step (1 frame)
        self.F = np.array([
            [1, 0, 0, 0, dt, 0,  0],   # cx = cx + vx*dt
            [0, 1, 0, 0, 0,  dt, 0],   # cy = cy + vy*dt  
            [0, 0, 1, 0, 0,  0,  0],   # aspect_ratio = aspect_ratio (constant)
            [0, 0, 0, 1, 0,  0,  dt],  # h = h + vh*dt
            [0, 0, 0, 0, 1,  0,  0],   # vx = vx
            [0, 0, 0, 0, 0,  1,  0],   # vy = vy
            [0, 0, 0, 0, 0,  0,  1]    # vh = vh
        ], dtype=np.float32)
        
        # Measurement matrix (we observe center position and size)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # measure cx
            [0, 1, 0, 0, 0, 0, 0],  # measure cy
            [0, 0, 1, 0, 0, 0, 0],  # measure aspect_ratio
            [0, 0, 0, 1, 0, 0, 0]   # measure height
        ], dtype=np.float32)
        
        # Process noise covariance - Standard ByteTrack parameters
        self.Q = np.eye(7, dtype=np.float32)
        self.Q[:2, :2] *= 1.0      # Position noise
        self.Q[2, 2] *= 1e-2       # Aspect ratio is more stable
        self.Q[3, 3] *= 1.0        # Height noise
        self.Q[4:, 4:] *= 1e-1     # Velocity noise
        
        # Measurement noise covariance - Detection uncertainty
        self.R = np.eye(4, dtype=np.float32)
        self.R[:2, :2] *= 1.0      # Position measurement noise
        self.R[2, 2] *= 10.0       # Aspect ratio measurement noise
        self.R[3, 3] *= 1.0        # Height measurement noise
        
        # Initial state covariance
        self.P = np.eye(7, dtype=np.float32)
        self.P[:2, :2] *= 10.0     # Initial position uncertainty
        self.P[2, 2] *= 10.0       # Initial aspect ratio uncertainty
        self.P[3, 3] *= 10.0       # Initial height uncertainty
        self.P[4:, 4:] *= 1000.0   # Very uncertain about initial velocities
        
    def predict(self):
        """Predict next state using motion model"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_bbox()
    
    def update(self, bbox):
        """Update with new detection in [cx, cy, w, h] format"""
        # bbox is already in center format [cx, cy, w, h]
        cx = bbox[0]  # Already center x
        cy = bbox[1]  # Already center y
        w, h = bbox[2], bbox[3]
        aspect_ratio = w / h if h != 0 else 1.0
        
        z = np.array([[cx], [cy], [aspect_ratio], [h]], dtype=np.float32)
        
        # Kalman update equations
        y = z - self.H @ self.x  # Innovation (residual)
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        # Update state and covariance
        self.x = self.x + K @ y
        I = np.eye(7)
        self.P = (I - K @ self.H) @ self.P
        
        return self.get_bbox()
    
    def get_bbox(self):
        """Return state back to bbox format [cx, cy, w, h]"""
        cx, cy, aspect_ratio, h = self.x[:4, 0]
        
        # Convert back to width
        w = aspect_ratio * h
        
        return np.array([cx, cy, w, h])