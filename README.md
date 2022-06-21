# Introduction #
This project allows one to simulate a random walker that moves randomly over a pre-defined n-dimensional grid while plotting its trajectory, calculating the return probability for each step and monitoring and plotting the spectral dimension.

# Dependencies #

Numpy and matplotlib.

# Usage #
The file `random_walker.py` contains two classes: `Walker` and `WalkerAnalyzer`. The first one includes the random walker and allows recording and plotting its trajectory, as well as defining basis vectors for the grid and playing around with the number of steps to be performed.

## Creating a random walker ##
To create a random walker (assuming the class was imported correctly), use the following syntax:
```python
w = Walker(nstep, dim, resting_allowed=False, orthonormal_basis=True)
```
**Parameters**
- `nstep`: Integer, number of steps to be performed in each simulation
- `dim`: Integer, dimension of the grid (min: 1)
- `resting_allowed`: Boolean, allow or forbid the walker to choose to not take a step (same probability as each of the other directions)
- `orthonormal_basis`: Boolean, initialize the walker with an orthonormal basis (i.e. a square grid) with dimension `dim`. If False, you have to define your own basis before running a simulation (see below)

## Define basis ##
The basis must be a Python list containing `dim` basis vectors in the form of numpy arrays of length `dim`. Example:
```python
w = Walker(nstep=30, dim=2, orthonormal_basis=False)
basis = [np.array([1.0, 1.0], np.array([1.0, -1.0]))]
w.basis = basis
```

## Get trajectory (run simulation) ##
To run a simulation and to get the walker's trajectory, use the method `trajectory`. The method returns a numpy array with shape (`nstep + 1`, `dim`), so each row corresponds to the walker's position at a given timestep (where the first row is always the origin). Example:
```python
w = Walker(nstep=30, dim=2)
trajectory = w.trajectory()
```

## Plot trajectory ##
Assuming you already have a trajectory, you can plot it using the method `plot`. Note that this only works for `dim <= 2`. Example:
```python
w = Walker(nstep=30, dim=2)
trajectory = w.trajectory()

w.plot(trajectory)
```

## Compute return probability ##
The return probability of a timestep is the probability that the walker has returned to the origin in that timestep, so for `nstep` timesteps, you get `nstep + 1` return probabilities (including the 0th timestep). To compute the return probabilities, use the method `return_probability(nrepeat)` in the class `WalkerAnalyzer`. Because this is a stochastic process, the method automatically runs the simulation `nrepeat` times. Note that, before creating a `WalkerAnalyzer` object, a walker must be created and the basis must be defined (either manually or by setting `orthonormal_basis=True`. Example:
```python
w = Walker(nstep=30, dim=2)
wa = WalkerAnalyzer(walker)

return_probability = wa.return_probability(nrepeat=10000)
```

## Plot return probability ##
To plot the return probabilities, use the method `plot_probability` in the class `WalkerAnalyzer`. Example:
```python
w = Walker(nstep=30, dim=2)
wa = WalkerAnalyzer(walker)

return_probability = wa.return_probability(nrepeat=10000)
wa.plot_probability(return_probability)
```

## Compute spectral dimension  ##
To compute the walker's spectral dimension, use the method `spectral_dimension` in the class `Walker Analyzer`. This returns a numpy array of length `nstep + 1` with the spectral dimension for each timestep (including the 0th step). Example:
```python
w = Walker(nstep=30, dim=2)
wa = WalkerAnalyzer(walker)

return_probability = wa.return_probability(nrepeat=10000)
spectral_dimension = wa.spectral_dimension(return_probability)
```

## Plot spectral dimension ##
To plot the walker's spectral dimension, use the method `plot_spectral_dimension` in the class `Walker Analyzer`. Example:
```python
w = Walker(nstep=30, dim=2)
wa = WalkerAnalyzer(walker)

return_probability = wa.return_probability(nrepeat=10000)
spectral_dimension = wa.spectral_dimension(return_probability)

wa.plot_spectral_dimension(spectral_dimension)
```

# Contribution #
As this program was made as a university assignment, the project is not being maintained, so no contribution is possible.