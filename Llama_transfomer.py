import jax
import jax.numpy as jnp


# x, weights, cos, sin are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    weights: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    seq_len: int,
) -> jax.Array:
    # return output tensor directly
    epsilon = 1e-5
    w1 = weights[:512]

    WQ = weights[512:262656]
    WQ = jax.numpy.reshape(a=WQ, shape=(512,512))

    WK = weights[262656:328192]
    WK = jax.numpy.reshape(a=WK, shape=(128,512))

    WV = weights[328192:393728]
    WV = jax.numpy.reshape(a=WV, shape=(128,512))

    WO = weights[393728:655872]
    WO = jax.numpy.reshape(a=WO, shape=(512,512))

    w2 = weights[655872:656384]

    WG = weights[656384:1377280]
    WG = jax.numpy.reshape(a=WG, shape=(1408,512))

    WU = weights[1377280:2098176]
    WU = jax.numpy.reshape(a=WU, shape=(1408,512))

    WD = weights[2098176:2819072]
    WD = jax.numpy.reshape(a=WD, shape=(512,1408))

    #RMSNorm 1
    rmsnorm1_denom = jnp.sqrt((jnp.mean(x ** 2, axis=-1, keepdims=True) + epsilon))
    rmsnorm1 = x / rmsnorm1_denom
    rmsnorm1 = rmsnorm1 * w1

    #QKV Projection
    Q = jnp.matmul(rmsnorm1,WQ.T)
    Q = jax.numpy.reshape(a=Q, shape=(-1,8,64))

    K = jnp.matmul(rmsnorm1,WK.T)
    K = jax.numpy.reshape(a=K, shape=(-1,2,64))
    K = jnp.repeat(K, 4, axis=1)
    
    V = jnp.matmul(rmsnorm1,WV.T)
    V = jax.numpy.reshape(a=V, shape=(-1,2,64))
    V = jnp.repeat(V, 4, axis=1)
    
    def rope(q,cos,sin):
        cos = cos[:,None,:]
        sin = sin[:,None,:]
        q1 = q[... ,:32]; q2 = q[... ,32:]
        m1 = ((q1 * cos) - (q2 * sin))
        m2 = ((q1 * sin) + (q2 * cos))
        return jnp.concatenate([m1,m2],axis = -1)

    Q = rope(Q,cos,sin)
    K = rope(K,cos,sin)

    Q = Q.swapaxes(0, 1)
    K = K.swapaxes(0, 1)
    V = V.swapaxes(0, 1)

    c_mask = jnp.triu(jnp.ones((x.shape[0],x.shape[0])),k=1) * -1e9
    print(c_mask.shape)

    heads = jnp.matmul(jax.nn.softmax(((jnp.matmul(Q,K.swapaxes(-1,-2)))/jnp.sqrt(64)) + c_mask),V)
    attention = jnp.matmul(heads.swapaxes(0, 1).reshape(x.shape[0], 512), WO.T)

    x_delta = attention + x

    #RMSNorm2
    rmsnorm2_denom = jnp.sqrt((jnp.mean(x_delta ** 2, axis=-1, keepdims=True) + epsilon))
    rmsnorm2 = x_delta / rmsnorm2_denom
    rmsnorm2 = rmsnorm2 * w2

    #FFN
    #Silu
    silu = jax.nn.silu((jnp.matmul(rmsnorm2,WG.T)))

    up_proj = rmsnorm2 @ WU.T

    su = silu * up_proj
    # Down
    ffn = su @ WD.T

    output = ffn + x_delta
    
    return output
