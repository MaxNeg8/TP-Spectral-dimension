import numpy as np
from matplotlib import pyplot as plt
import time

class BasisError(Exception):
        pass

class DimensionError(Exception):
        pass

class Walker:

    def __init__(self, nstep, dim, resting_allowed=False, orthonormal_basis=True):
        self.nstep = nstep
        self.dim = dim
        self.resting_allowed = resting_allowed

        self.__basis = []

        if orthonormal_basis:
            self.__orthonormal_basis()

    @property
    def basis(self):
        return self.__basis
    
    @basis.setter
    def basis(self, basis):
        if isinstance(basis, list):
            if len(basis) == self.dim:
                for v in basis:
                    if not isinstance(v, np.ndarray):
                        raise BasisError("Basis must be list of numpy float arrays")
                    if v.dtype != float:
                        raise BasisError("Basis must be list of numpy float arrays")
                    if v.shape != (self.dim,):
                        raise BasisError(f"Basis vectors must be of shape ({self.dim},) for a {self.dim} dimensional walker")
            else:
                raise BasisError(f"Basis must be a list of length {self.dim} for a {self.dim} dimensional walker")
        else:
            raise BasisError("Basis must be a list of basis vectors")
        if np.abs(np.linalg.det(np.array(basis))) <= 1e-5:
            raise BasisError("Basis mut be linearly independent")
        self.__basis = basis

    def __orthonormal_basis(self):
        b = np.eye(self.dim)
        self.basis = [i for i in b]
            
    def __walk(self, position):

        if len(self.basis) == 0:
            raise BasisError("Basis must be defined before walker can walk")

        if self.resting_allowed:
            if np.random.rand() <= 1/(2*self.dim + 1):
                return position.copy()

        sign = np.random.choice([-1, 1])
        return position.copy() + sign * self.basis[np.random.randint(0, self.dim)]
        

    def trajectory(self):
        trajectory = np.zeros(shape=(self.nstep+1, self.dim), dtype=float)

        for t in range(1, self.nstep+1):
            trajectory[t] = self.__walk(trajectory[t-1])
        
        return trajectory

    def plot(self, trajectory):

        if self.dim == 2:
            fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=100)

            ax.scatter(trajectory[:, 0], trajectory[:, 1])
            ax.plot(trajectory[:, 0], trajectory[:, 1])

            ax.set_xlabel("x-Axis")
            ax.set_ylabel("y-Axis")
            ax.set_title(f"Plot of walker's trajectory (resting_allowed={self.resting_allowed}, timestamp={time.time()})")

            plt.show()
        elif self.dim == 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 3), dpi=100)

            ax.scatter(trajectory[:, 0], np.zeros(self.nstep+1))
            ax.plot(trajectory[:, 0], np.zeros(self.nstep+1))

            ax.set_xlabel("x-Axis")
            ax.set_title(f"Plot of walker's trajectory (resting_allowed={self.resting_allowed}, timestamp={time.time()})")
            
            ax.get_yaxis().set_visible(False)

            plt.show()
        else:
            raise DimensionError("Dimension of the walker must be 1 or 2 in order to plot its trajectory")


class WalkerAnalyzer:

    def __init__(self, walker):
        self.walker = walker
        if len(self.walker.basis) == 0:
            raise BasisError("Walker must have defined basis to be analyzed")
    
    def return_probability(self, nrepeat):
        returned = np.zeros(self.walker.nstep+1, dtype=int)
        for i in range(nrepeat):
            trajectory = self.walker.trajectory()
            for i, position in enumerate(trajectory):
                if np.linalg.norm(position) < 1e-5:
                    returned[i] += 1
        return returned/nrepeat

    def plot_probability(self, probability):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        ax.scatter(np.arange(0, self.walker.nstep+1, 1), probability)
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Probability")
        ax.set_title(f"Return probability for random walker (timestamp={time.time()})")
        
        plt.show()

    def spectral_dimension(self, probability):
        diff = np.zeros(len(probability))
        for t in range(1, len(probability)-1):
            diff[t] = t/probability[t] * (probability[t + 1] - probability[t - 1])
        diff[len(probability)-1] = 2*(len(probability)-1)/(probability[len(probability)-1]) * (probability[len(probability)-1] - probability[len(probability)-2])
        return -1 * diff

    def plot_spectral_dimension(self, spectral_dimension):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        ax.scatter(np.arange(0, self.walker.nstep+1, 1), spectral_dimension)
        ax.plot(np.arange(0, self.walker.nstep+1, 1), spectral_dimension)
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Spectral dimension")
        ax.set_title(f"Spectral dimension for random walker (timestamp={time.time()})")
        
        plt.show()


def main():
    w1 = Walker(30, 4, resting_allowed=True, orthonormal_basis=True)
    wa1 = WalkerAnalyzer(w1)
    wa1.plot_spectral_dimension(wa1.spectral_dimension(wa1.return_probability(int(1e6))))
    


if __name__ == "__main__":
    main()