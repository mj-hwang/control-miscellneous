import numpy as np

class LTIEnv(object):
    def __init__(self, A, B, Q, R, x_init):
        self.A = A
        self.B = B
        self.x_dim = self.A.shape[0]
        self.u_dim = self.B.shape[1]
        self.Q = Q
        self.R = R
        self.x_init = x_init
        self.x_curr = x_init
        self.t = 0
        
    def controllable(self):
        # controllability matrix
        C = [self.B]
        for i in range(self.x_dim):
            C.append(self.A @ C[-1])
        return np.linalg.matrix_rank(np.hstack(C)) == x_dim

    def step(self, u, noise=None):
        if noise is not None:
            assert noise.shape == self.x_curr.shape
            x = self.A @ self.x_curr + self.B @ u + noise
        else:
            x = self.A @ self.x_curr + self.B @ u
        c = self.x_curr.T @ self.Q @ self.x_curr + u.T @ self.R @ u # cost
        self.x_curr = x.copy()
        self.t += 1
        return x, c

    def get_state(self):
        return self.x_curr.copy()
    
    def set_state(self, x):
        self.x_curr = x.copy()

    def reset(self):
        self.t = 0
        self.x_curr = self.x_init.copy()
        
class LTVEnv(object):
    def __init__(self, A_lst, B_lst, Q, R, x_init):
        self.A_lst = A_lst
        self.B_lst = B_lst
        self.x_dim = self.A_lst[0].shape[0]
        self.u_dim = self.B_lst[0].shape[1]
        self.Q = Q
        self.R = R
        self.x_init = x_init
        self.x_curr = x_init
        # self.T = len(self.A_lst)
        self.t = 0

    def step(self, u, noise=None):
        if noise is not None:
            assert noise.shape == self.x_curr.shape
            x = self.A_lst[self.t] @ self.x_curr + self.B_lst[self.t] @ u + noise
        else:
            x = self.A_lst[self.t] @ self.x_curr + self.B_lst[self.t] @ u
        c = self.x_curr.T @ self.Q @ self.x_curr + u.T @ self.R @ u # cost
        self.x_curr = x.copy()
        self.t += 1
        return x, c

    def get_state(self):
        return self.x_curr.copy()
    
    def set_state(self, x):
        self.x_curr = x.copy()

    def reset(self):
        self.t = 0
        self.x_curr = self.x_init.copy()
        
class NonLinearEnv(object):
    def __init__(self, f, Q, R, x_init, target_traj=None):
        self.f = f
        self.Q = Q
        self.R = R
        self.x_dim = self.Q.shape[0]
        self.u_dim = self.R.shape[0]
        
        self.target_traj = target_traj
        
        self.x_init = x_init
        self.x_curr = x_init
        self.t = 0

    def step(self, u, noise=None):
        if noise is not None:
            assert noise.shape == self.x_curr.shape
            x = f(self.x_curr, u) + noise
        else:
            x = f(self.x_curr, u)
            
        if self.target_traj is not None:
            x_target = target_traj[0][t]
            u_target = target_traj[1][t]
    
            c = (self.x_curr-x_target).T @ self.Q @ (self.x_curr-x_target) + \
                (u-u_target).T @ self.R @ (u-u_target) # cost         
        else:
            c = self.x_curr.T @ self.Q @ self.x_curr + u.T @ self.R @ u # cost
        self.x_curr = x.copy()
        self.t += 1
        return x, c

    def get_state(self):
        return self.x_curr.copy()
    
    def set_state(self, x):
        self.x_curr = x.copy()
        
    def reset(self):
        self.t = 0
        self.x_curr = self.x_init.copy()
        
    def linearize_dynamics(f, x_ref, u_ref, dt, eps):
        A, B = np.zeros((x_dim, x_dim)), np.zeros((x_dim, u_dim))
        
        for i in range(dx):
            delta_x = np.zeros(dx)
            delta_x[i] = eps
            A[:, i] = (f(x_ref+delta_x, u_ref, dt) - f(x_ref, u_ref, dt)) / eps

        for j in range(du):
            delta_u = np.zeros(du)
            delta_u[j] = eps
            B[:, j] = (f(x_ref, u_ref+delta_u, dt) - f(x_ref, u_ref, dt)) / eps

        c = f(x_ref, u_ref, dt) - x_ref_next
        
        return A, B, c