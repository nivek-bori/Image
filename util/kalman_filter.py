import numpy as np


class KalmanFilter:
    def __init__(self, bbox):
        cx, cy, w, h = bbox

        # x: [cx, cy, width, height, vx, vy, vw, vh]
        # velocity is initally zero
        self.x = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)

        # State transition matrix (constant velocity model)
        dt = 1  # Time step (1 frame)
        self.F = np.array(
            [
                [1, 0, 0, 0, dt, 0, 0, 0],  # cx = cx + vx*dt
                [0, 1, 0, 0, 0, dt, 0, 0],  # cy = cy + vy*dt
                [0, 0, 1, 0, 0, 0, dt, 0],  # width = width + vw*dt
                [0, 0, 0, 1, 0, 0, 0, dt],  # height = height + vh*dt
                [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
                [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
                [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
                [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
            ],
            dtype=np.float32,
        )
        dt2 = 0.9 # For slower steps
        self.F2 = np.array(
            [
                [1, 0, 0, 0, dt, 0, 0, 0],  # cx = cx + vx*dt
                [0, 1, 0, 0, 0, dt, 0, 0],  # cy = cy + vy*dt
                [0, 0, 1, 0, 0, 0, dt2, 0],  # width = width + vw*dt
                [0, 0, 0, 1, 0, 0, 0, dt2],  # height = height + vh*dt
                [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
                [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
                [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
                [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
            ],
            dtype=np.float32,
        )

        # Measurement matrix (we observe center position and size)
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],  # measure cx
                [0, 1, 0, 0, 0, 0, 0, 0],  # measure cy
                [0, 0, 1, 0, 0, 0, 0, 0],  # measure width
                [0, 0, 0, 1, 0, 0, 0, 0],  # measure height
            ],
            dtype=np.float32,
        )

        ###
        # Q: how unpredictable a value is. In practice, how much can the value change
        # R: how much to trust detection measurements. In practice, how much detection measurements can affect predictions.
        # P: how uncertain new tracking is. In practice, how volitile inital updates can be
        ###

        # Process noise covariance - Standard ByteTrack parameters
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[:2, :2] *= 1.0  # Position noise
        self.Q[2, 2] *= 1.0  # Width noise
        self.Q[3, 3] *= 1.0  # Height noise
        self.Q[4:, 4:] *= 0.1  # Velocity noise

        # Measurement noise covariance - Detection uncertainty
        self.R = np.eye(4, dtype=np.float32)
        self.R[:2, :2] *= 1000.0  # Position measurement noise
        self.R[2, 2] *= 1000.0  # Width measurement noise
        self.R[3, 3] *= 1000.0  # Height measurement noise

        # Initial state covariance
        self.P = np.eye(8, dtype=np.float32)
        self.P[:2, :2] *= 0.0  # Initial position uncertainty
        self.P[2, 2] *= 30.0  # Initial width uncertainty
        self.P[3, 3] *= 30.0  # Initial height uncertainty
        self.P[4:, 4:] *= 1000 * 1000.0  # Very uncertain about initial velocities

    # steps internal states
    def predict(self, slower=False):
        # is it possible to slow down the speed that width and height change when k_filter is predicting w/out data?
        # if slower:
        #     self.x = self.F2 @ self.x
        # else:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_cxywh()

    # doesn't step internal states
    def predict_n_steps(self, steps, stride=1):
        pred_x = self.x.copy()
        pred_P = self.P.copy()

        predicted_states = [pred_x.copy()]

        for step in range(steps):
            # step by stride num
            for _i in range(stride):
                pred_x = self.F @ pred_x
                pred_P = self.F @ pred_P @ self.F.T + self.Q

            predicted_states.append(pred_x.copy())

        return predicted_states

    def update(self, bbox):
        cx, cy, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)

        # Kalman update equations
        y = z - self.H @ self.x  # Innovation (residual)
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        # Update state and covariance
        self.x = self.x + K @ y
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P

        return self.get_cxywh()

    def get_cxywh(self):
        cx, cy, w, h = self.x[:4, 0]
        return np.array([cx, cy, w, h])  # [cx, cy, w, h]

    def get_xywh(self):
        cx, cy, w, h = self.x[:4, 0]
        x, y = cx - w / 2, cy - h / 2
        return np.array([x, y, w, h])
