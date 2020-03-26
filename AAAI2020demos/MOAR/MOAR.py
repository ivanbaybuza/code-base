from functions import *
from tensorly.base import unfold
import pickle as pkl

# I1, I2, T = 6, 8, 4
# J1, J2 = 3, 4
#
# X = np.arange(I1 * I2 * T).reshape((I1, I2, T))
#
# U = random_init([I1, I2], [J1, J2])
#
# Y = []

__all__ = ("MOAR")

def MOAR(X, core=None, m=3, max_iter_num=20):

    T = X.shape[-1]
    dims = X[..., 0].shape
    # U = random_init(list(dims), core)
    # U = init(list(dims), core)
    U = svd_init(X[..., 0], list(range(len(dims))), core)

    Y = []
    for it in range(max_iter_num):

        old_U = U.copy()

        for t in range(T):
            X_t = X[..., t]
            modes = list(range(len(X_t.shape)))
            Y_t = multi_mode_dot(X_t, U, modes=modes)
            Y.append(Y_t)

        ar = fit_ar(Y, p=m)

        E_h = []
        for t in range(m, T):
            temp = None
            for k in range(m):
                temp = ar[k] * X[..., t-m+k] if temp is None else temp + ar[k] * X[..., t-m+k]
            E_h.append(X[..., t] - temp)


        for i in range(len(U)):
            modes = list(range(len(X.shape) - 1))
            modes.remove(i)
            # Eq. 18
            Fi = None
            for t in range(m, T):
                Fi_t = unfold(multi_mode_dot(E_h[t-m], [U[dim] for dim in modes], modes), i)
                Fi = Fi_t.dot(Fi_t.T) if Fi is None else Fi + Fi_t.dot(Fi_t.T)
            _, uj = np.linalg.eig(Fi)

            U[i] = uj[:, :U[i].shape[0]].T

        delta_U = []
        for u1, u2 in zip(old_U, U):
            delta_U.append(u1 - u2)
        delta_norm = sum_frobenius_norm(delta_U)
        new_norm = sum_frobenius_norm(U)
        print('U change:', delta_norm / new_norm)

    return U, ar


if __name__ == '__main__':

    m = 2
    max_iter_num = 100

    with open('ushcn.npy', 'rb') as file:
        data = pkl.load(file)

    num_ts = 120
    tr_stop = int(95 * 0.9)
    data = data[:, :num_ts, 0, :1]  # (95, 120,  12, 4)
    data = data.transpose((1, 2, 0))
    traindata = data[..., :tr_stop]
    testdata = data[..., tr_stop-m:]
    label = data[..., tr_stop:]

    U, ar = MOAR(X=traindata, core=[50, 1], m=m, max_iter_num=max_iter_num)

    # Predict:
    predictions = np.stack(predict(U, testdata, ar), axis=2)

    # ACC = get_acc(predictions[:, :], label[:, :])
    # print(ACC)

    ERR = get_error(label, predictions)

    print(ERR)
