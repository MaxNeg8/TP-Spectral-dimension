import numpy as np
from matplotlib import pyplot as plt
import time

#Exception classes for use in Walker and WalkerAnalyzer

class BasisError(Exception):
        pass

class DimensionError(Exception):
        pass

#Class for random walker
class Walker:

    def __init__(self, nstep, dim, resting_allowed=False, orthonormal_basis=True):
        """
        Initializer for Walker class.

        Parameters:
            nstep (int): Number of steps to take in each simulation
            dim (int): Dimension of the grid (min: 1)
            resting_allowed (boolean): Allow walker to rest (same probability as walking)
            orthonormal_basis (boolean): Initialize walker with orthonormal basis of dimension dim
        """
        self.nstep = nstep
        self.dim = dim
        self.resting_allowed = resting_allowed

        self.__basis = []

        if orthonormal_basis:
            #Initialize with orthonormal basis
            self.__orthonormal_basis()

    @property
    def basis(self):
        """
        Getter for basis.

        Returns:
            basis (list): List of basis vectors (numpy.ndarray) of length dim
        """
        return self.__basis
    
    @basis.setter
    def basis(self, basis):
        """
        Setter for basis.

        Parameters:
            basis (list): List of basis vectors (numpy.ndarray) of length dim
        """
        if isinstance(basis, list):
            if len(basis) == self.dim:
                for v in basis:
                    #Check vectors in basis
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
        self.__basis = basis

    def __orthonormal_basis(self):
        """
        Helper function that initializes the walker's basis as an orthonormal one with dimension dim
        """
        b = np.eye(self.dim)
        self.basis = [i for i in b]
            
    def __walk(self, position):
        """
        Helper function, lets the walker take one random step (or not, depending on value of resting_allowed)

        Paramters:
            position (numpy.ndarray): The walker's current position

        Returns:
            New position of the walker
        """

        #Check whether basis was defined
        if len(self.basis) == 0:
            raise BasisError("Basis must be defined before walker can walk")

        if self.resting_allowed:
            #If resting is allowed, let the walker not take a step with the proper probability of 1/(2*dim + 1)
            if np.random.rand() <= 1/(2*self.dim + 1):
                return position.copy()

        #Direction along the chosen basis vector
        sign = np.random.choice([-1, 1])
        
        #Update position with random basis vector
        return position.copy() + sign * self.basis[np.random.randint(0, self.dim)]
        

    def trajectory(self):
        """
        Compute the walker's trajectory for nstep steps.

        Returns:
            trajectory (numpy.ndarray): Matrix with shape (nstep+1, dim) where each row is the walker's position at a specific timestep
        """
        trajectory = np.zeros(shape=(self.nstep+1, self.dim), dtype=float)

        #Walk nstep times and record positions in trajectory array
        for t in range(1, self.nstep+1):
            trajectory[t] = self.__walk(trajectory[t-1])
        
        return trajectory

    def plot(self, trajectory):
        """
        Plot the walker's trajectory (only works if dim <= 2).

        Parameters:
            trajectory (numpy.ndarray): Walker's trajectory obtained from method trajectory()
        """

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

#Class that allows a random walker's trajectory to be analyzed
class WalkerAnalyzer:

    def __init__(self, walker):
        #Initializer
        self.walker = walker
        if len(self.walker.basis) == 0:
            raise BasisError("Walker must have defined basis to be analyzed")
    
    def return_probability(self, nrepeat):
        """
        Compute the return probability for each timestep (including 0th one).

        Parameters:
            nrepeat (int): Number of times the simulation is run

        Returns:
            probability (numpy.ndarray): Array containing the return probability for each timestep
        """
        returned = np.zeros(self.walker.nstep+1, dtype=int)
        #Repeat process nrepeat times
        for i in range(nrepeat):
            #Compute trajectory
            trajectory = self.walker.trajectory()
            for i, position in enumerate(trajectory):
                #If walker has returned to origin, increase number of returns for this timestep in returned array
                if np.linalg.norm(position) < 1e-5:
                    returned[i] += 1
        #Compute probabilities by dividing number of returns for each timestep by the number of tries
        return returned/nrepeat

    def plot_probability(self, probability):
        """
        Allows the return probabilities obtained from method return_probability() to be plotted.

        Parameters:
            probability (numpy.ndarray): Return probabilities from method return_probability()
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        ax.scatter(np.arange(0, self.walker.nstep+1, 1), probability)
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Probability")
        ax.set_title(f"Return probability for random walker (timestamp={time.time()})")
        
        plt.show()

    def spectral_dimension(self, probability):
        """
        Compute the spectral dimension for the random walker.

        Parameters:
            probability (numpy.ndarray): Return probabilities obtained from method return_probability()

        Returns:
            spectral_dimension (numpy.ndarray): Spectral dimension for each timestep
        """
        diff = np.zeros(len(probability))
        #Use intermediate difference to compute spectral dimension for each timestep (Skips 0th timestep as the spectral dimension here is always 0)
        for t in range(1, len(probability)-1):
            diff[t] = t/probability[t] * (probability[t + 1] - probability[t - 1])
        #Use backward difference to compute spectral dimension for the last timestep (as intermediate difference cannot be used)
        diff[len(probability)-1] = 2*(len(probability)-1)/(probability[len(probability)-1]) * (probability[len(probability)-1] - probability[len(probability)-2])
        return -1 * diff

    def plot_spectral_dimension(self, spectral_dimension):
        """
        Allows the spectral dimension obtained from method spectral_dimension() to be plotted.

        Parameters:
            spectral_dimension (numpy.ndarray): Spectral dimension obtained from method spectral_dimension()
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        ax.scatter(np.arange(0, self.walker.nstep+1, 1), spectral_dimension)
        ax.plot(np.arange(0, self.walker.nstep+1, 1), spectral_dimension)
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Spectral dimension")
        ax.set_title(f"Spectral dimension for random walker (timestamp={time.time()})")
        
        plt.show()


def main():
    #Example random walker
    w = Walker(30, 2, resting_allowed=True, orthonormal_basis=True)
    wa = WalkerAnalyzer(w)
    probability = wa.return_probability(int(1e6))
    wa.plot_probability(probability)
    wa.plot_spectral_dimension(wa.spectral_dimension(probability))
    


if __name__ == "__main__":
    main()