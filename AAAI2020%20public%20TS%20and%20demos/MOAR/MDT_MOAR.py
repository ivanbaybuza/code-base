from functions import *
from tensorly.base import unfold
import pickle as pkl
from MDT import MDTWrapper
from metric import *
import time

class MDT_MOAR(object):

    def __init__(self, X, taus, Rs, m=3, max_iter_num=20, init_svd=True, threshold=0.7):
        self._X = X
        self._Rs = Rs
        self._m = m
        self._taus = taus
        self._max_iter_num = max_iter_num
        self._init_svd = init_svd
        self._threshold = threshold
        self._T = X.shape[-1]
        self._ts_ori_shape = X.shape

    def _forward_MDT(self, data, taus):
        self.mdt = MDTWrapper(data, taus)
        trans_data = self.mdt.transform()
        self._T_hat = self.mdt.shape()[-1]
        print("the shape after MDT:", trans_data.shape)
        return trans_data, self.mdt

    def _inverse_MDT(self, mdt, data, taus, shape):
        return mdt.inverse(data, taus, shape)

    def _get_cores(self, tensor, Us):
        T_hat = tensor.shape[-1]
        cores = [tl.tenalg.multi_mode_dot(tensor[..., t], [u for u in Us], modes=[i for i in range(len(Us))]) for t in range(T_hat)]
        return cores

    def _initilize_U(self, trans_data, Rs):
        T_hat = trans_data.shape[-1]
        dim = trans_data.ndim
        factors = svd_init(trans_data[..., 0], range(len(trans_data.shape)-1), ranks=Rs)

        return factors

    def _initializer(self, trans_data, Rs):

        T_hat = trans_data.shape[-1]
        # initilize Us
        if self._init_svd:
            U = self._initilize_U(trans_data, Rs)
        else:
            U = [np.random.random([j, r]) for j, r in zip(list(trans_data.shape)[:-1], Rs)]
        self._U = U

        return self._U


    def run(self):
        X_hat, mdt = self._forward_MDT(self._X, self._taus)
        T_hat = X_hat.shape[-1]
        Us = self._initializer(X_hat, self._Rs)

        for it in range(max_iter_num):
            print("Iteration:\t", it)
            old_U = Us.copy()
            Y = self._get_cores(X_hat, Us)
            ar = fit_ar(Y, p=self._m)

            # Calculate E:
            E_h = []
            for t in range(m, T_hat):
                temp = None
                for k in range(m):
                    temp = ar[k] * X_hat[..., t - m + k] if temp is None else temp + ar[k] * X_hat[..., t - m + k]
                E_h.append(X_hat[..., t] - temp)

            # Update Us:
            for i in range(len(Us)):
                modes = list(range(len(X_hat.shape) - 1))
                modes.remove(i)
                # Eq. 18
                Fi = None
                for t in range(m, T_hat):
                    Fi_t = unfold(multi_mode_dot(E_h[t - m], [Us[dim] for dim in modes], modes), i)
                    Fi = Fi_t.dot(Fi_t.T) if Fi is None else Fi + Fi_t.dot(Fi_t.T)
                _, uj = np.linalg.eig(Fi)

                Us[i] = uj[:, :Us[i].shape[0]].T

            delta_U = []
            for u1, u2 in zip(old_U, Us):
                delta_U.append(u1 - u2)
            delta_norm = sum_frobenius_norm(delta_U)
            new_norm = sum_frobenius_norm(Us)
            # print('U change:', delta_norm / new_norm)
            if delta_norm / new_norm < self._threshold:
                print("Early stop at iteration ", it)
                break

        # Forecasting:
        Y = self._get_cores(X_hat, Us)

        new_core = np.sum([al * core for al, core in zip(ar, Y[-self._m:])], axis=0)
        mdt_result = tl.tenalg.multi_mode_dot(new_core, [u.T for u in Us])

        # Inverse MDT
        fore_shape = list(self._ts_ori_shape)

        merged = []
        for i in range(X_hat.shape[-1]):
            merged.append(X_hat[..., i].T)
        merged.append(mdt_result.T)
        merged = np.array(merged)
        mdt_result = merged.T

        fore_shape[-1] += 1
        fore_shape = np.array(fore_shape)

        result = self._inverse_MDT(mdt, mdt_result, self._taus, fore_shape)

        return result

if __name__ == '__main__':

    m = 3
    max_iter_num = 30
    taus = np.array([607, 4, 5])
    Rs = [20, 3, 5]

    data = load_data('dataset/D1.npy')
    # data = data[:, :120, 6]
    data[1:, :, 0] = data[:-1, :, 0]

    data = data[1:, :, :]

    T = data.shape[0]
    # data = data.T
    data = np.transpose(data, (1, 2, 0))

    print(data.shape)

    predictions = []
    labels = []

    sta = time.time()

    round = 0
    for tr_stop in range(int(T * 0.9), T, 1):
        round += 1
        traindata = data[..., :tr_stop]
        model = MDT_MOAR(X=traindata, taus=taus, Rs=Rs, m=m, max_iter_num=max_iter_num)
        pred = model.run()
        predictions.append(pred[:, 0, -1])
        labels.append(data[:, 0, tr_stop])

    end = time.time()

    print("Time Consumption: ", (end-sta) / round)

    predictions = np.stack(predictions)
    labels = np.stack(labels)

    eval_forecast(np.real(predictions).flatten(), labels.flatten())
