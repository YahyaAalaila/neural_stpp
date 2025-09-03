class PiecewiseConstantControl:
    """
    z_values: (N, T, Dz).  z_i applies over (t_i, t_{i+1}] and last value over (t_T, t1].
    """
    def __init__(self, z_values):
        self.z_values = z_values
    def at_interval(self, i):
        return self.z_values[:, i]  # (N, Dz)
