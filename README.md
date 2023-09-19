# QAOA

This package is a versatile python implementation of the [Quantum Approximate Optimization Algorithm](https://arxiv.org/pdf/1411.4028.pdf) /[Quantum Alternating Operator Ansatz](https://arxiv.org/pdf/1709.03489.pdf)  (QAOA) aimed for researchers to quickly test the performance of new ansätze, new classical optimizers, etc.


***
### Background
Given a **cost function** 
$$c: \{ 0, 1\}^n \rightarrow \mathbb{R}$$
one defines a **problem Hamiltonian** $H_P$ through the action on computational basis states via
$$ H_P |x\rangle = c(x) |x\rangle,$$
which means that ground states minimize the cost function $c$.
Given a parametrized ansatz $ | \gamma, \beta \rangle$, a classical optimizers tries than to minimize the energy
$$ \langle \gamma, \beta | H_P | \gamma, \beta \rangle$$.

***
### Ansatz
QAOA consist of the following **ansatz**:

$$ |\gamma, \beta \rangle = \prod_{l=1}^p \left( U_M(\beta_l) U_P(\gamma_l)\right) | s\rangle, $$

where

- $U_P$ is a family of **phase**-separating operators,
- $U_M$ is a family of **mixing** operators, and
- $|s\rangle$ is a "simple" **initial** state.

Typicall these have the form
$U_M(\beta_l)=e^{-i\beta_l H_M}$,  and $U_P(\gamma_l)=e^{-i\gamma_l H_P}$, and the simplest initial state is the uniform superposition, i.e. $| s \rangle = |+\rangle^{\otimes n}$. 

***
### API of this library

This library mimicks how one makes an ansatz by specifying classes that implement

- an initial state,
- a problem instance, and
- a mixer

		qaoamc = QAOA(
			initialstate=initialstates.Plus(),
			problem=problems.MaxCut(G=G),
			mixer=mixers.X()
		)
