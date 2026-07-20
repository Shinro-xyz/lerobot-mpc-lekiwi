import pytest
import numpy as np
from utils.array_backend import NumpyBackend


class TestNumpyBackend:
    def setup_method(self):
        self.bk = NumpyBackend()

    def test_array(self):
        a = self.bk.array([1, 2, 3])
        assert isinstance(a, np.ndarray)
        assert a.dtype == np.float64
        assert np.allclose(a, [1, 2, 3])

    def test_zeros(self):
        z = self.bk.zeros(2, 3)
        assert z.shape == (2, 3)
        assert np.all(z == 0)

    def test_zeros_like(self):
        x = np.array([[1, 2], [3, 4]], dtype=float)
        z = self.bk.zeros_like(x)
        assert z.shape == (2, 2)
        assert np.all(z == 0)

    def test_eye(self):
        I = self.bk.eye(3)
        assert I.shape == (3, 3)
        assert np.allclose(I, np.eye(3))

    def test_diag(self):
        d = self.bk.diag([1, 2, 3])
        assert d.shape == (3, 3)
        assert np.allclose(d, np.diag([1, 2, 3]))

    def test_inv(self):
        A = np.array([[4, 7], [2, 6]], dtype=float)
        Ainv = self.bk.inv(A)
        assert np.allclose(A @ Ainv, np.eye(2))

    def test_pinv(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        Apinv = self.bk.pinv(A)
        assert np.allclose(A @ Apinv @ A, A)

    def test_solve(self):
        A = np.array([[3, 1], [1, 2]], dtype=float)
        b = np.array([9, 8], dtype=float)
        x = self.bk.solve(A, b)
        assert np.allclose(A @ x, b)

    def test_norm(self):
        v = np.array([3, 4], dtype=float)
        assert np.allclose(self.bk.norm(v), 5.0)

    def test_cross(self):
        a = np.array([1, 0, 0], dtype=float)
        b = np.array([0, 1, 0], dtype=float)
        c = self.bk.cross(a, b)
        assert np.allclose(c, [0, 0, 1])

    def test_sin_cos_arccos(self):
        theta = np.pi / 4
        s = self.bk.sin(theta)
        c = self.bk.cos(theta)
        assert np.allclose(s, np.sqrt(2) / 2)
        assert np.allclose(c, np.sqrt(2) / 2)
        assert np.allclose(self.bk.arccos(c), theta)

    def test_trace(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        assert self.bk.trace(A) == 5.0

    def test_clip(self):
        x = np.array([-1, 0.5, 2], dtype=float)
        clipped = self.bk.clip(x, 0.0, 1.0)
        assert np.allclose(clipped, [0, 0.5, 1])

    def test_where(self):
        cond = np.array([True, False, True])
        a = np.array([1, 2, 3], dtype=float)
        b = np.array([10, 20, 30], dtype=float)
        result = self.bk.where(cond, a, b)
        assert np.allclose(result, [1, 20, 3])

    def test_any(self):
        assert self.bk.any(np.array([False, True]))
        assert not self.bk.any(np.array([False, False]))

    def test_copy(self):
        x = np.array([1, 2, 3], dtype=float)
        y = self.bk.copy(x)
        y[0] = 99
        assert x[0] == 1

    def test_kron(self):
        a = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([[0, 5], [6, 7]], dtype=float)
        k = self.bk.kron(a, b)
        assert k.shape == (4, 4)

    def test_eigvals(self):
        A = np.array([[1, 0], [0, 2]], dtype=float)
        eigs = self.bk.eigvals(A)
        assert np.allclose(np.sort(eigs), [1, 2])

    def test_matrix_rank(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        assert self.bk.matrix_rank(A) == 1
        B = np.eye(3)
        assert self.bk.matrix_rank(B) == 3

    def test_cond(self):
        A = np.eye(3)
        assert np.allclose(self.bk.cond(A), 1.0)

    def test_svd(self):
        A = np.array([[1, 0], [0, 2], [0, 0]], dtype=float)
        U, s, Vh = self.bk.svd(A)
        assert U.shape[0] == 3
        assert len(s) == 2
        assert Vh.shape[1] == 2
        assert np.allclose(U[:, :2] @ np.diag(s) @ Vh, A)

    def test_real(self):
        x = np.array([1 + 2j, 3 + 4j])
        assert np.allclose(self.bk.real(x), [1, 3])

    def test_sort(self):
        x = np.array([3, 1, 2], dtype=float)
        assert np.allclose(self.bk.sort(x), [1, 2, 3])

    def test_sqrt(self):
        x = np.array([4, 9, 16], dtype=float)
        assert np.allclose(self.bk.sqrt(x), [2, 3, 4])

    def test_abs(self):
        x = np.array([-1, -2, 3], dtype=float)
        assert np.allclose(self.bk.abs(x), [1, 2, 3])

    def test_sum(self):
        x = np.array([1, 2, 3, 4], dtype=float)
        assert self.bk.sum(x) == 10.0

    def test_reshape(self):
        x = np.array([1, 2, 3, 4], dtype=float)
        y = self.bk.reshape(x, 2, 2)
        assert y.shape == (2, 2)

    def test_ravel(self):
        x = np.array([[1, 2], [3, 4]], dtype=float)
        y = self.bk.ravel(x)
        assert y.shape == (4,)

    def test_linspace(self):
        x = self.bk.linspace(0, 1, 5)
        assert len(x) == 5
        assert np.allclose(x[0], 0)
        assert np.allclose(x[-1], 1)

    def test_matrix_power(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        A2 = self.bk.matrix_power(A, 2)
        assert np.allclose(A2, A @ A)

    def test_cholesky(self):
        A = np.array([[4, 2], [2, 3]], dtype=float)
        L = self.bk.cholesky(A)
        assert np.allclose(L @ L.T, A)

    def test_vstack(self):
        a = np.array([1, 2], dtype=float)
        b = np.array([3, 4], dtype=float)
        s = self.bk.vstack([a, b])
        assert s.shape == (2, 2)

    def test_hstack(self):
        a = np.array([[1], [2]], dtype=float)
        b = np.array([[3], [4]], dtype=float)
        s = self.bk.hstack([a, b])
        assert s.shape == (2, 2)

    def test_block(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5], [6]], dtype=float)
        C = np.array([[7, 8]], dtype=float)
        D = np.array([[9]], dtype=float)
        M = self.bk.block([[A, B], [C, D]])
        assert M.shape == (3, 3)

    def test_tile(self):
        x = np.array([1, 2], dtype=float)
        t = self.bk.tile(x, 3)
        assert np.allclose(t, [1, 2, 1, 2, 1, 2])

    def test_to_numpy(self):
        x = np.array([1, 2, 3], dtype=float)
        assert self.bk.to_numpy(x) is x

    def test_from_numpy(self):
        x = np.array([1, 2, 3], dtype=float)
        assert self.bk.from_numpy(x) is x


class TestTorchBackend:
    def setup_method(self):
        torch = pytest.importorskip("torch")
        from utils.array_backend import TorchBackend
        self.bk = TorchBackend(device="cpu")
        self.torch = torch

    def test_array(self):
        a = self.bk.array([1, 2, 3])
        assert isinstance(a, self.torch.Tensor)
        assert a.dtype == self.torch.float64

    def test_zeros(self):
        z = self.bk.zeros(2, 3)
        assert z.shape == (2, 3)
        assert self.bk.to_numpy(self.bk.sum(z)) == 0

    def test_eye(self):
        I = self.bk.eye(3)
        assert I.shape == (3, 3)
        assert self.bk.to_numpy(self.bk.allclose(I, self.torch.eye(3)))

    def test_inv(self):
        A = self.bk.array([[4, 7], [2, 6]])
        Ainv = self.bk.inv(A)
        assert self.bk.to_numpy(self.bk.allclose(A @ Ainv, self.torch.eye(2)))

    def test_svd(self):
        A = self.bk.array([[1, 0], [0, 2], [0, 0]])
        U, s, Vh = self.bk.svd(A)
        assert U.shape[0] == 3
        assert len(s) == 2
        assert Vh.shape[1] == 2

    def test_cholesky(self):
        A = self.bk.array([[4, 2], [2, 3]])
        L = self.bk.cholesky(A)
        assert self.bk.to_numpy(self.bk.allclose(L @ L.T, A))

    def test_to_numpy_roundtrip(self):
        x = np.array([1, 2, 3], dtype=float)
        y = self.bk.from_numpy(x)
        z = self.bk.to_numpy(y)
        assert np.allclose(x, z)

    def allclose(self, a, b):
        return self.torch.allclose(a, b, atol=1e-8)
