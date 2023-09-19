# QAOA

This package is a flexible python implementation of the [Quantum Approximate Optimization Algorithm](https://arxiv.org/pdf/1411.4028.pdf) /[Quantum Alternating Operator Ansatz](https://arxiv.org/pdf/1709.03489.pdf)  (QAOA) aimed for researchers to quickly test the performance of new ansätze, new classical optimizers, etc. By default it uses qiskit as a backend.


***
### Background
Given a **cost function** 
$$c: \{ 0, 1\}^n \rightarrow \mathbb{R}$$
one defines a **problem Hamiltonian** $H_P$ through the action on computational basis states via

$$ H_P |x\rangle = c(x) |x\rangle,$$

which means that ground states minimize the cost function $c$.
Given a parametrized ansatz $ | \gamma, \beta \rangle$, a classical optimizer is used to minimize the energy

$$ \langle \gamma, \beta | H_P | \gamma, \beta \rangle.$$

***
### Ansatz
QAOA consist of the following **ansatz**:

$$ |\gamma, \beta \rangle = \prod_{l=1}^p \left( U_M(\beta_l) U_P(\gamma_l)\right) | s\rangle, $$

where

- $U_P$ is a family of **phase**-separating operators,
- $U_M$ is a family of **mixing** operators, and
- $|s\rangle$ is a "simple" **initial** state.

Typicall these have the form
$U_M(\beta_l)=e^{-i\beta_l H_M}$,  $U_P(\gamma_l)=e^{-i\gamma_l H_P}$, and the simplest initial state is the uniform superposition, i.e. $| s \rangle = |+\rangle^{\otimes n}$. 

***
### Basic API of this library

This library mimicks how one makes an ansatz by specifying classes that implement (with the following 

- a [mixer](qaoa/mixers/base_mixer.py) with implementations:
	- [X-mixer](qaoa/mixers/x_mixer.py),
	- [XY-mixer](qaoa/mixers/xy_mixer.py),
	- [Grover-mixer](qaoa/mixers/grover_mixer.py),
- a  [problem](qaoa/problems/base_problem.py) with implementations:
	- [maxcut](qaoa/problems/maxcut_problem.py)
	- [QUBO](qaoa/problems/qubo_problem.py),
	- [Exact cover](qaoa/problems/exactcover_problem.py),
	- [Portfolio](qaoa/problems/portfolio_problem.py),
- an [initial state](qaoa/initialstates/base_initialstate.py) with implementations:
	- [Plus](qaoa/initialstates/plus_initialstate.py),
	- [Statevector](qaoa/initialstates/statevector_initialstate.py),
	- [Dicke](qaoa/initialstates/dicke_initialstate.py).

The base classes have an `@abstractmethod` called `create_circuit`which needs to be implemented.
The problem base class additionally has an `@abstractmethod` called `cost`.

To make a concrete ansatz, one can create an instance like this: 

		qaoamc = QAOA(
			initialstate=initialstates.Plus(),
			problem=problems.MaxCut(G=[networkx instance]),
			mixer=mixers.X()
		)

Of course, one can easily create more instances, by providing a differet implementation of the base classes.

***
### Further parameters
