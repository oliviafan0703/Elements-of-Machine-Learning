import autograd.numpy as np
from decorators import CallCounter


def reshape_if_needed(z, samples, m=2, d=4):
    if z.size == m:
        z = np.reshape(z, (m, 1))
    else:
        assert z.ndim > 1 and z.shape[0] == m, 'first dimension of z must be {}'.format(m)
    if samples.size == d:
        samples = np.expand_dims(samples.flatten(), axis=0)
    else:
        assert samples.ndim == 2 and samples.shape[1] == d,\
            'samples must be {}-dimensional'.format(d)
    output_reps = samples.shape[0]
    return z, samples, output_reps


def outer(a, b):
    return np.einsum('a,b...->ab...', a, b)


def broadcast(a, b):
    return np.einsum('a,a...->a...', a, b)


@CallCounter
def phi(z, samples):
    z, samples, reps = reshape_if_needed(z, samples)
    amps = samples[:, ::2]
    omegas = samples[:, 1::2]
    x_min = 2.
    b = 0.2

    def gp(u):
        dx = u - x_min
        xp = u  + broadcast(0.01 * amps[:, 0], np.sin(outer(omegas[:, 0], dx)))
        q = np.sin(3. * np.pi * xp) * np.sin(np.pi * (xp - 2.) ** 2.)
        p = q + broadcast(amps[:, 1], np.sin(outer(omegas[:, 1], dx)))
        return p

    y_min = gp(np.array([x_min]))[0][0]
    phi.z_ast = np.array((x_min, y_min))

    x, y = z[0], z[1]
    x_rep, y_rep = outer(np.ones(reps), x), outer(np.ones(reps), y)
    v = (y_rep - gp(x)) ** 2 + b * (x_rep - x_min) ** 2
    return v


def f(z, samples):
    value = np.mean(phi(z, samples), axis=0)
    if value.size == 1:
        value = value[0]
    return value
