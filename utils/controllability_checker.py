import numpy as np
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov
from scipy.sparse import issparse, csc_matrix, csr_matrix, isspmatrix_csc, isspmatrix_csr


class LTISystemsAnalyzer:
    def __init__(self, A:np.ndarray, B:np.ndarray= None, C:np.ndarray=None, D:np.ndarray= None, dt:float=None):
        self.A=A #state matrix
        self.B=B if B is not None else np.zeros((A.shape[0],0)) #input matrix
        self.C=C if C is not None else np.zeros((0,A.shape[0])) #output matrix
        self.D = D if D is not None else np.zeros((self.C.shape[0],self.B.shape[1])) #feedthrough matrix
        self.dt=dt
        self._cached_values={}

    def __post_init__(self) -> None:
        """Validate dimensions and fill missing matrices with zeros."""
        n = self.A.shape[0]

        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square (n×n).")

        # Helper to create a zero matrix with matching sparsity
        def _zero(shape):
            if issparse(self.A):
                return csr_matrix(shape, dtype=float)
            return np.zeros(shape, dtype=float)

        self.B = _zero((n, 0)) if self.B is None else self.B
        self.C = _zero((0, n)) if self.C is None else self.C
        self.D = _zero(self.C.shape[:1] + self.B.shape[1:]) if self.D is None else self.D

        # sanity check dimensions
        if self.B.shape[0] != n:
            raise ValueError("B must have same row count as A.")
        if self.C.shape[1] != n:
            raise ValueError("C must have same column count as A.")
        if self.D.shape != (self.C.shape[0], self.B.shape[1]):
            raise ValueError("D must be (p×m) consistent with C and B.")

    
    def controllabilty(self):
        #verifying the inputs A and B
        
        n=self.A.shape[0]
    
        #constructing the controllability matrix
    
        cols=[self.B]
        for i in range(1,n):
            cols.append(self.A@cols[-1])
    
        C=np.hstack(cols)
    
        # check the rank of the matrix and controllability
        rank=np.linalg.matrix_rank(C)
        is_controllable=(rank==n)
        return C
    
    def observability(self):
        n= self.A.shape[0]
        cols=[self.C]
    
        for i in range(1,n):
            cols.append(self.C@np.linalg.matrix_power(self.A,i))
    
        O=np.vstack(cols) #stacking the list vertically
    
        rank= np.linalg.matrix_rank(O)
        is_observable=(rank==n)
    
        return O

    def _rank_and_conidtion_check(self,M:np.ndarray):
        rank=np.linalg.matrix_rank(M)
        cond= np.linalg.cond(M) if rank==M.shape[0] else np.inf
        return rank, cond

    def is_controllable(self):
        rank,_=self.controllabilty()
        return rank==self.A.shape[0]

    def is_observable(self):
      rank,_= self.observability()
      return rank==self.A.shape[0]

    # Continuous Time Gramian calculations

    def _solve_continuous_lyap(self,Q:np.ndarray):
        return solve_continuous_lyapunov(self.A,-Q)

    def controllability_gramian(self) -> np.ndarray:
        """
        Infinite‑horizon controllability Gramian (continuous‑time).

        Returns cached result unless the system has changed.  Raises
        `ValueError` if A is not Hurwitz (unstable) because the integral diverges.
        """
        if "Wc" in self._cached_values:
            return self._cached_values["Wc"]

        # Quick Hurwitz test: eigenvalues must have negative real parts
        eigA = np.linalg.eigvals(self.A)
        if np.any(np.real(eigA) >= 0):
            raise ValueError(
                "A is not Hurwitz; infinite‑horizon controllability Gramian does not exist. "
                "Use `controllability_gramian_finite(T)` instead."
            )
        Q = self.B @ self.B.T
        Wc = self._solve_continuous_lyap(Q)
        self._cached_values["Wc"] = Wc
        return Wc

    def observability_gramian(self) -> np.ndarray:
        """
        Infinite‑horizon observability Gramian (continuous‑time).

        Equivalent checks as `controllability_gramian`.
        """
        if "Wo" in self._cached_values:
            return self._cached_values["Wo"]

        eigA = np.linalg.eigvals(self.A)
        if np.any(np.real(eigA) >= 0):
            raise ValueError(
                "A is not Hurwitz; infinite‑horizon observability Gramian does not exist. "
                "Use `observability_gramian_finite(T)` instead."
            )
        Q = self.C.T @ self.C
        Wo = self._solve_continuous_lyap(Q)
        self._cached_values["Wo"] = Wo
        return Wo
    
    # Discrete Time Gramians, using Discrete Lyapunov equations
    
    def _solve_discrete_lyap(self,M: np.ndarray):
        return solve_discrete_lyapunov(self.A, -M)

    def discrete_controllability_gramian(self):
        if self.dt is None:
            raise ValueError("dt is None, use controllability_gramian() for continuous functions, set dt>0 for discrete tiem systems")
            