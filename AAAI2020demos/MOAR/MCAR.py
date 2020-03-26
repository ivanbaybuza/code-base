from functions import *
from tensorly.base import unfold
import matplotlib.pyplot as plt
import pickle as pkl

# I1, I2, T = 6, 8, 80
# J1, J2 = 3, 3
#
# X = np.random.rand(I1 * I2 * T).reshape((I1, I2, T))
#
# U = random_init([I1, I2], [J1, J2])



def MCAR(X, core=None, m=3, fi=0.1, iterations=20):

    T = X.shape[-1]
    dims = X[..., 0].shape
    U = random_init(list(dims), core)
    delta = []

    for it in range(iterations):

        # old_norm = sum_frobenius_norm(U)

        Y = []
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

            Fi = None
            Gi = None

            for t in range(m, T):
                Fi_t = unfold(multi_mode_dot(E_h[t-m], [U[dim] for dim in modes], modes), mode=i)
                Gi_t = unfold(multi_mode_dot(X[..., t], [U[dim] for dim in modes], modes), mode=i)

                temp1 = Fi_t.dot(Fi_t.T)
                temp2 = np.dot(unfold(Y[t], mode=i), Gi_t.T) - U[i].dot(Gi_t).dot(Gi_t.T)

                if Fi is not None:
                    Fi += temp1
                    Gi += temp2
                else:
                    Fi = temp1
                    Gi = temp2

            # Eq. 27
            U[i] = 2 * fi * linalg.pinv(Fi).dot(Gi.T).T

        # new_norm = sum_frobenius_norm(U)
    #     delta.append(new_norm - old_norm)
    #
    # plt.plot(delta)
    # plt.show()
    return U, ar


if __name__ =='__main__':

    m = 3
    fi = 0.1
    iterations = 10

    with open('ts.npy', 'rb') as file:
        data = pkl.load(file)

    data = data.transpose((1, 2, 0))
    traindata = data[:, :, :34]
    testdata = data[:, :, 34-m:]
    label = data[:, :, 34:]

    U, ar = MCAR(X=traindata, core=[20, 3], m=3, fi=fi, iterations=10)

    # Predict:
    predictions = np.stack(predict(U, testdata, ar), axis=2)

    ACC = get_acc(predictions[:, :], label[:, :])
    print(ACC)
