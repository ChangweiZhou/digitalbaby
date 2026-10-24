import numpy as np
def learn_sr(gamma=0.8, passes=500, lr=0.1, seed=0):
    M=np.eye(10)
    for _ in range(passes):
        seq=list(range(10))
        for s,s2 in zip(seq[:-1],seq[1:]):
            M[s]+=lr*(np.eye(10)[s]+gamma*M[s2]-M[s])
        M[9]+=lr*(np.eye(10)[9]-M[9])          # end of stream
    return M
SR=learn_sr()
SRs=(SR+SR.T)/2                                  # forward + backward exposure (symmetric context)
