import numpy as np
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov

class LTISystemsAnalyzer:
    def __init__(self, A:np.ndarray, B:np.ndarray= None, C:np.ndarray=None, D:np.ndarray= None, dt:float=None):
        self.A=A #state matrix
        self.B=B if B is not None else np.zeros((A.shape[0],0)) #input matrix
        self.C=C if C is not None else np.zeros((0,A.shape[0])) #output matrix
        self.D = D if D is not None else np.zeros((self.C.shape[0],self.B.shape[1])) #feedthrough matrix
        self.dt=dt
        self._cached_values={}

    def __post_init__(self)->None:
        n=self.A.shape[0] #how many states?
        if self.A.shape[0]!=self.A.shape[1]:
            raise ValueError("A must be a square matrix")

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

    def _solve_continuous_lyapunov(self,Q:np.ndarray):
        return solve_continuous_lyapunov(self.A,-Q)

    def controllability_gramian(self) -> np.ndarray:
        """
        Infinite‑horizon controllability Gramian (continuous‑time).

        Returns cached result unless the system has changed.  Raises
        `ValueError` if A is not Hurwitz (unstable) because the integral diverges.
        """
        if "Wc" in self._cache:
            return self._cache["Wc"]

        # Quick Hurwitz test: eigenvalues must have negative real parts
        eigA = eigvals(self.A)
        if np.any(np.real(eigA) >= 0):
            raise ValueError(
                "A is not Hurwitz; infinite‑horizon controllability Gramian does not exist. "
                "Use `controllability_gramian_finite(T)` instead."
            )
        Q = self.B @ self.B.T
        Wc = self._solve_continuous_lyap(Q)
        self._cache["Wc"] = Wc
        return Wc

    def observability_gramian(self) -> np.ndarray:
        """
        Infinite‑horizon observability Gramian (continuous‑time).

        Equivalent checks as `controllability_gramian`.
        """
        if "Wo" in self._cache:
            return self._cache["Wo"]

        eigA = eigvals(self.A)
        if np.any(np.real(eigA) >= 0):
            raise ValueError(
                "A is not Hurwitz; infinite‑horizon observability Gramian does not exist. "
                "Use `observability_gramian_finite(T)` instead."
            )
        Q = self.C.T @ self.C
        Wo = self._solve_continuous_lyap(Q)
        self._cache["Wo"] = Wo
        return Wo