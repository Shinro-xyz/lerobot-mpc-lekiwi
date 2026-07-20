import numpy as np
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov, sqrtm
from scipy.sparse import issparse, csc_matrix, csr_matrix, isspmatrix_csc, isspmatrix_csr
from scipy.integrate import solve_ivp



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
    
    def _solve_discrete_lyap(self,A,M: np.ndarray):

        eigs=np.linalg.eigvals(self.A)
        if np.any(np.abs(eigs)>=1.0):
            raise ValueError(
                        "Discrete‑time system is not asymptotically stable (|eig| >= 1). "
                        "The infinite‑horizon Gramian does not exist. "
                        "Use a finite‑horizon Gramian instead."
                    )

        if self.dt is None:
            raise ValueError("dt is None, use controllability_gramian() for continuous functions, set dt>0 for discrete tiem systems")

        return solve_discrete_lyapunov(A,M)

    def discrete_controllability_gramian(self):
        if "Wc_discrete" in self._cached_values:
            return self._cached_values["Wc_discrete"]

        M=self.B@self.B.T
        Wc_discrete=self._solve_discrete_lyap(self.A,M)
        self._cached_values["Wc_discrete"]=Wc_discrete
        return Wc_discrete

    def discrete_observability_gramian(self):
        if "Wo_discrete" in self._cached_values:
            return self._cached_values["Wo_discrete"]

        M=self.C.T@self.C
        Wo_discrete=self._solve_discrete_lyap(self.A.T,M)
        self._cached_values["Wo_discrete"]=Wo_discrete
        return Wo_discrete

    # Finite Time Gramians-> both observability and controllability

    def _gramian_ode(self, t, w_flat, Q, A_mat):
        """dW/dt = A W + W Aᵀ + Q."""
        n = A_mat.shape[0]
        W = w_flat.reshape((n, n))
        dW = A_mat @ W + W @ A_mat.T + Q
        return dW.ravel()

    def _finite_horizon_gramian(
        self,
        Q: np.ndarray,
        T: float,
        method: str = "RK45",
        num_pts: int = 500,
    ) -> np.ndarray:
        """Common routine for finite‑horizon Gramians."""
        n = self.A.shape[0]
        A_mat = self.A.toarray() if issparse(self.A) else self.A
        sol = solve_ivp(
            fun=self._gramian_ode,
            t_span=(0.0, T),
            y0=np.zeros(n * n),
            args=(Q, A_mat),
            method=method,
            t_eval=np.linspace(0.0, T, num_pts),
        )
        if not sol.success:
            raise RuntimeError("IVP integration for Gramian failed.")
        return sol.y[:, -1].reshape((n, n))

    def controllability_gramian_finite(self, T: float) -> np.ndarray:
        """
        Finite‑horizon controllability Gramian:
            Wc(T) = ∫₀ᵀ e^{Aτ} B Bᵀ e^{Aᵀτ} dτ

        Works for any A (stable or unstable).  `T` must be > 0.
        """
        if T <= 0:
            raise ValueError("T (horizon) must be positive.")
        Q = self.B @ self.B.T
        return self._finite_horizon_gramian(Q, T)

    def observability_gramian_finite(self, T: float) -> np.ndarray:
        """
        Finite‑horizon observability Gramian:
            Wo(T) = ∫₀ᵀ e^{Aᵀτ} Cᵀ C e^{Aτ} dτ
        """
        if T <= 0:
            raise ValueError("T (horizon) must be positive.")
        Q = self.C.T @ self.C
        return self._finite_horizon_gramian(Q, T)

  # ------------------------------------------------------------------
    # 6.  Spectral diagnostics (eigenvalues, singular values, condition)
    # ------------------------------------------------------------------
    def gramian_spectrum(self, gramian: str = "Wc") -> np.ndarray:
        """
        Return eigenvalues of a chosen Gramian.

        Parameters
        ----------
        gramian : {"Wc", "Wo", "Wc_finite", "Wo_finite"}
            Which Gramian to inspect.
        """
        if gramian == "Wc":
            G = self.controllability_gramian()
        elif gramian == "Wo":
            G = self.observability_gramian()
        elif gramian.startswith("Wc_"):
            # expect format "Wc_finite:5.0"
            _, horizon = gramian.split(":")
            G = self.controllability_gramian_finite(float(horizon))
        elif gramian.startswith("Wo_"):
            _, horizon = gramian.split(":")
            G = self.observability_gramian_finite(float(horizon))
        else:
            raise ValueError(f"Unknown gramian identifier: {gramian}")

        return np.real(np.linalg.eigvals(G))

    def gramian_condition(self, gramian: str = "Wc") -> float:
        """
        Return the 2‑norm condition number of the selected Gramian.
        """
        if gramian == "Wc":
            G = self.controllability_gramian()
        elif gramian == "Wo":
            G = self.observability_gramian()
        else:
            raise ValueError("Only infinite‑horizon 'Wc' / 'Wo' supported for cond().")
        # Condition is infinite if the matrix is singular
        rank = np.linalg.matrix_rank(G)
        if rank < G.shape[0]:
            return np.inf
        return np.linalg.cond(G)

    # ------------------------------------------------------------------
    # 7.  Hankel singular values & balanced realization
    # ------------------------------------------------------------------
    def hankel_singular_values(self) -> np.ndarray:
        """
        Compute the Hankel singular values σ_i = sqrt(λ_i(Wc Wo)).
        Returns a sorted (descending) array of length n.
        """
        Wc = self.controllability_gramian()
        Wo = self.observability_gramian()
        prod = Wc @ Wo
        # Eigenvalues of the product may be complex because of round‑off;
        # we keep the real part and take the sqrt.
        eigs = np.real(np.linalg.eigvals(prod))
        eigs[eigs < 0] = 0.0                     # clip negatives caused by numerical noise
        sigma = np.sqrt(np.sort(eigs)[::-1])
        return sigma

    def balanced_realization(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (Abal, Bbal, Cbal) – the balanced state‑space matrices.
        The transformation T satisfies:
            T⁻¹ A T = Abal,
            T⁻¹ B   = Bbal,
            C T    = Cbal,
        and the balanced Gramians are diag(σ₁,…,σₙ).
        """
        sigma = self.hankel_singular_values()
        # Form the square‑root diagonal matrix Σ½
        Sigma_sqrt = np.diag(np.sqrt(sigma))

        # Compute the Cholesky factors of the Gramians (they are PD)
        Wc = self.controllability_gramian()
        Wo = self.observability_gramian()
        # Take the symmetric square root (more stable than Cholesky for near‑singular)
        Lc = sqrtm(Wc)
        Lo = sqrtm(Wo)

        # Singular value decomposition of Loᵀ Lc
        U, s, Vh = np.linalg.svd(Lo.T @ Lc)
        T = Lc @ Vh.T @ np.diag(1.0 / np.sqrt(s)) @ U.T @ Lo.T

        # Apply the transformation
        Ab = np.linalg.inv(T) @ self.A @ T
        Bb = np.linalg.inv(T) @ self.B
        Cb = self.C @ T

        return Ab, Bb, Cb

    # ------------------------------------------------------------------
    # 8.  Balanced truncation (model reduction)
    # ------------------------------------------------------------------
    def balanced_truncate(self, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform balanced truncation to obtain an order‑r reduced model.

        Returns (Ar, Br, Cr, Dr) where Dr = D (unchanged).

        Parameters
        ----------
        r : int
            Desired reduced order (0 < r <= n).
        """
        n = self.A.shape[0]
        if not (0 < r <= n):
            raise ValueError("Reduced order r must satisfy 0 < r ≤ n.")

        # Get balanced matrices and Hankel values
        Ab, Bb, Cb = self.balanced_realization()
        sigma = self.hankel_singular_values()

        # Partition according to r
        Ar = Ab[:r, :r]
        Br = Bb[:r, :]
        Cr = Cb[:, :r]
        # Dr stays unchanged
        Dr = self.D

        # (Optional) compute an a‑posteriori error bound: 2 * sum_{i=r+1}^{n} σ_i
        error_bound = 2.0 * np.sum(sigma[r:])
        self._cached_values["balanced_trunc_error_bound"] = error_bound

        return Ar, Br, Cr, Dr

    # ------------------------------------------------------------------
    # 9.  Utility / housekeeping
    # ------------------------------------------------------------------
    def reset_cache(self) -> None:
        """Clear all memoised results – useful after you manually change A, B or C."""
        self._cached_values.clear()

    def summary(self) -> str:
        """
        Produce a short, human‑readable report that covers the most useful
        quantities: rank/cond, gramian eigenvalues, Hankel values, and a
        quick verdict on controllability/observability.
        """
        n = self.A.shape[0]
        rank_info = self.rank_report()
        wc = self.controllability_gramian() if self.is_controllable() else None
        wo = self.observability_gramian() if self.is_observable() else None

        lines = [
            f"System order n = {n}",
            f"Controllable?      {self.is_controllable()}",
            f"Observable?       {self.is_observable()}",
            "",
            "Kalman rank / condition:",
            f"  Controllability : rank = {rank_info['controllability'][0]},  cond = {rank_info['controllability'][1]:.2e}",
            f"  Observability   : rank = {rank_info['observability'][0]},    cond = {rank_info['observability'][1]:.2e}",
            "",
        ]

        if wc is not None:
            eig_wc = np.real(np.linalg.eigvals(wc))
            lines.append(f"Controllability Gramian eigenvalues (sorted):")
            lines.append("  " + ", ".join(f"{ev:.3e}" for ev in np.sort(eig_wc)[::-1]))
        else:
            lines.append("Controllability Gramian: *not defined* (unstable A).")

        if wo is not None:
            eig_wo = np.real(np.linalg.eigvals(wo))
            lines.append(f"Observability Gramian eigenvalues (sorted):")
            lines.append("  " + ", ".join(f"{ev:.3e}" for ev in np.sort(eig_wo)[::-1]))
        else:
            lines.append("Observability Gramian: *not defined* (unstable A).")

        # Hankel singular values
        sigma = self.hankel_singular_values()
        lines.append("")
        lines.append("Hankel singular values (σ₁ ≥ σ₂ …):")
        lines.append("  " + ", ".join(f"{sv:.3e}" for sv in sigma))

        # Optional reduced‑order error bound (if we have computed before)
        err = self._cached_values.get("balanced_trunc_error_bound")
        if err is not None:
            lines.append("")
            lines.append(f"Balanced‑truncation error bound (2·∑σ_{r+1..n}) = {err:.3e}")

        return "\n".join(lines)