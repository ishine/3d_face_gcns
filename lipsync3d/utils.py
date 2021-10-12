import numpy as np

# Input :
#       reference(dictionary from vertex idx to normalized landmark, dict[idx] = [x, y, z]) : landmark of reference frame.
#       target(dictionary from vertex idx to normalized landmark, dict[idx] = [x, y, z]) : landmark of target frame.
# Output : 
#       R : 3x3 Rotation matrix(np.array)
#       c : scale value(float)
#       t : 3x1 translation matrix(np.array)

def Umeyama_algorithm(reference, target):
    # idx 2 -> nose, 130 -> left eye, 359 -> right eye
    idx_list = [2, 94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 9, 151, 10, 109, 108, 67, 69, 103, 104, 54, 68, 338, 337, 297, 299, 332, 333, 284, 298, 130, 243, 244, 359, 362, 463,
                21, 71, 162, 139, 156, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 301, 251, 55, 285, 193, 417, 122, 351, 196, 419, 3, 248, 51, 281,
                45, 275, 44, 274, 220, 440, 134, 363, 236, 456]
    # idx_list = [19, 243, 463]
    ref_points = []
    tgt_points = []

    for idx in idx_list:
        ref_points.append(reference[idx])
        tgt_points.append(target[idx])

    ref_points = np.array(ref_points)
    tgt_points = np.array(tgt_points)

    ref_mu = ref_points.mean(axis=0)
    tgt_mu = tgt_points.mean(axis=0)
    ref_var = ref_points.var(axis=0).sum()
    tgt_var = tgt_points.var(axis=0).sum()
    n, m = ref_points.shape
    covar = np.matmul((ref_points - ref_mu).T, tgt_points - tgt_mu) / n
    det_covar = np.linalg.det(covar)
    u, d, vh = np.linalg.svd(covar)
    detuv = np.linalg.det(u) * np.linalg.det(vh.T)
    cov_rank = np.linalg.matrix_rank(covar)
    S = np.identity(m)

    if cov_rank > m - 1:
        if det_covar < 0:
            S[m - 1, m - 1] = -1
    else: 
        if detuv < 0:
            S[m - 1, m - 1] = -1
    S = np.identity(m)
    R = np.matmul(np.matmul(u, S), vh)
    c = (1 / tgt_var) * np.trace(np.matmul(np.diag(d), S))
    t = ref_mu.reshape(3, 1) - c * np.matmul(R, tgt_mu.reshape(3, 1))

    return R, t, c

