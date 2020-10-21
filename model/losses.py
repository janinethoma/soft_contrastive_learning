import tensorflow as tf
import pointnetvlad_cls

# Our method
def wms_loss(distances, embeddings, d_alpha, d_beta, alpha=2.0, beta=50.0, lamb=1.0, eps=0.1, ms_mining=True, wfunction='exp', sumfunction='ms'):
  
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)

    batch_size = embeddings.get_shape().as_list()[0]

    if wfunction == 'lin':
        mask_pos = tf.where(distances < d_beta, 1.0 - tf.divide(distances, d_beta), tf.zeros_like(distances))
        mask_neg = tf.where(distances < d_beta, tf.divide(distances, d_beta), tf.ones_like(distances))
    elif wfunction == 'tanh':
        mask_pos = 1.0-tf.tanh(distances/d_beta)
        mask_neg = tf.tanh(distances/d_beta)
    else: # 'exp' default
        mask_pos = tf.divide(1.0, (1.0 + tf.exp(d_alpha * (distances - d_beta))))
        mask_neg = tf.divide(1.0, (1.0 + tf.exp(d_alpha * (d_beta - distances))))


    mask_pos = tf.cast(mask_pos, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32)
    mask_neg = tf.cast(mask_neg, dtype=tf.float32)

    sim_mat = tf.matmul(embeddings, embeddings, transpose_a=False, transpose_b=True)
    sim_mat = tf.maximum(sim_mat, 0.0)

    pos_mat = tf.multiply(sim_mat, mask_pos)
    neg_mat = tf.multiply(sim_mat, mask_neg)

    if ms_mining:
        max_val = tf.reduce_max(neg_mat, axis=1, keepdims=True)
        tmp_max_val = tf.reduce_max(pos_mat, axis=1, keepdims=True)
        min_val = tf.reduce_min(tf.multiply(sim_mat - tmp_max_val, mask_pos), axis=1, keepdims=True) + tmp_max_val

        mask_pos = tf.where(pos_mat < max_val + eps, mask_pos, tf.zeros_like(mask_pos))
        mask_neg = tf.where(neg_mat > min_val - eps, mask_neg, tf.zeros_like(mask_neg))

    if sumfunction == 'plain':
        pos_exp = tf.where(mask_pos > 0.0, pos_mat, tf.zeros_like(pos_mat))
        neg_exp = tf.where(mask_neg > 0.0, neg_mat, tf.zeros_like(neg_mat))

        pos_term = tf.reduce_sum(pos_exp, axis=1)
        neg_term = tf.reduce_sum(neg_exp, axis=1)

        loss = tf.reduce_mean(neg_term-pos_term)

    elif sumfunction == 'ms':
        pos_exp = tf.exp(-alpha * (pos_mat - lamb))
        pos_exp = tf.where(mask_pos > 0.0, pos_exp, tf.zeros_like(pos_exp))

        neg_exp = tf.exp(beta * (neg_mat - lamb))
        neg_exp = tf.where(mask_neg > 0.0, neg_exp, tf.zeros_like(neg_exp))

        pos_term = tf.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / alpha
        neg_term = tf.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / beta

        loss = tf.reduce_mean(pos_term + neg_term)

    return loss


def evil_triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
    worst_pos = worst_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg), 1])
    worst_pos = tf.tile(tf.reshape(worst_pos, (-1, 1)), [1, int(num_neg)])
    m = tf.fill([int(batch), int(num_neg)], margin)
    triplet_loss = tf.reduce_mean(tf.reduce_sum(
        tf.maximum(tf.add(m, tf.subtract(worst_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, query_copies), 2))),
                   tf.zeros([int(batch), int(num_neg)])), 1))
    return triplet_loss


def ms_loss(labels, embeddings, alpha=2.0, beta=50.0, lamb=1.0, eps=0.1, ms_mining=True):
    '''
    ref: http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    official codes: https://github.com/MalongTech/research-ms-loss
    '''

    # make sure emebedding should be l2-normalized
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    labels = tf.reshape(labels, [-1, 1])

    batch_size = embeddings.get_shape().as_list()[0]

    adjacency = tf.equal(labels, tf.transpose(labels))
    adjacency_not = tf.logical_not(adjacency)

    mask_pos = tf.cast(adjacency, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32)
    mask_neg = tf.cast(adjacency_not, dtype=tf.float32)

    sim_mat = tf.matmul(embeddings, embeddings, transpose_a=False, transpose_b=True)
    sim_mat = tf.maximum(sim_mat, 0.0)

    pos_mat = tf.multiply(sim_mat, mask_pos)
    neg_mat = tf.multiply(sim_mat, mask_neg)

    if ms_mining:
        max_val = tf.reduce_max(neg_mat, axis=1, keepdims=True)
        tmp_max_val = tf.reduce_max(pos_mat, axis=1, keepdims=True)
        min_val = tf.reduce_min(tf.multiply(sim_mat - tmp_max_val, mask_pos), axis=1, keepdims=True) + tmp_max_val

        max_val = tf.tile(max_val, [1, batch_size])
        min_val = tf.tile(min_val, [1, batch_size])

        mask_pos = tf.where(pos_mat < max_val + eps, mask_pos, tf.zeros_like(mask_pos))
        mask_neg = tf.where(neg_mat > min_val - eps, mask_neg, tf.zeros_like(mask_neg))

    pos_exp = tf.exp(-alpha * (pos_mat - lamb))
    pos_exp = tf.where(mask_pos > 0.0, pos_exp, tf.zeros_like(pos_exp))

    neg_exp = tf.exp(beta * (neg_mat - lamb))
    neg_exp = tf.where(mask_neg > 0.0, neg_exp, tf.zeros_like(neg_exp))

    pos_term = tf.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / alpha
    neg_term = tf.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / beta

    loss = tf.reduce_mean(pos_term + neg_term)

    return loss


def logratio_loss(a_feature, pos_features, neg_features, squared_pos_dists, squared_neg_dists):

    pos_residuals = tf.reduce_sum(tf.squared_difference(a_feature, pos_features),2)
    neg_residuals = tf.reduce_sum(tf.squared_difference(a_feature, neg_features),2)

    feat_ratio = tf.log(tf.div(pos_residuals, tf.transpose(neg_residuals)))
    dist_ratio = tf.log(tf.div(squared_pos_dists, tf.transpose(squared_neg_dists)))

    squared_diffs = tf.squared_difference(feat_ratio, dist_ratio)
    sum_of_squared_diffs = tf.reduce_mean(tf.reduce_mean(squared_diffs, 1),1)  # Mean over all positives and negatives
    return tf.reduce_mean(sum_of_squared_diffs, 0)  # Mean over batches



def ms_det(labels, embeddings, alpha=2.0, beta=50.0, lamb=1.0, eps=0.1, ms_mining=False):
    '''
    ref: http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    official codes: https://github.com/MalongTech/research-ms-loss
    '''

    # make sure emebedding should be l2-normalized
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    labels = tf.reshape(labels, [-1, 1])

    batch_size = embeddings.get_shape().as_list()[0]

    adjacency = tf.equal(labels, tf.transpose(labels))
    adjacency_not = tf.logical_not(adjacency)

    mask_pos = tf.cast(adjacency, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32)
    mask_neg = tf.cast(adjacency_not, dtype=tf.float32)

    sim_mat = tf.matmul(embeddings, embeddings, transpose_a=False, transpose_b=True)
    sim_mat = tf.maximum(sim_mat, 0.0)

    pos_mat = tf.multiply(sim_mat, mask_pos)
    neg_mat = tf.multiply(sim_mat, mask_neg)

    if ms_mining:
        max_val = tf.reduce_max(neg_mat, axis=1, keepdims=True)
        tmp_max_val = tf.reduce_max(pos_mat, axis=1, keepdims=True)
        min_val = tf.reduce_min(tf.multiply(sim_mat - tmp_max_val, mask_pos), axis=1, keepdims=True) + tmp_max_val

        max_val = tf.tile(max_val, [1, batch_size])
        min_val = tf.tile(min_val, [1, batch_size])

        mask_pos = tf.where(pos_mat < max_val + eps, mask_pos, tf.zeros_like(mask_pos))
        mask_neg = tf.where(neg_mat > min_val - eps, mask_neg, tf.zeros_like(mask_neg))

    pos_exp = tf.exp(-alpha * (pos_mat - lamb))
    pos_exp = tf.where(mask_pos > 0.0, pos_exp, tf.zeros_like(pos_exp))

    neg_exp = tf.exp(beta * (neg_mat - lamb))
    neg_exp = tf.where(mask_neg > 0.0, neg_exp, tf.zeros_like(neg_exp))

    pos_term = tf.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / alpha
    neg_term = tf.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / beta

    loss = tf.reduce_mean(pos_term + neg_term)

    return loss


def ms_sum(anchor, positives, negatives, margin, labels, embeddings, alpha=2.0, beta=50.0, lamb=1.0, eps=0.1,
           ms_mining=False, dimensions=10):
    ms = ms_loss(labels, embeddings, alpha, beta, lamb, eps, ms_mining)

    res = residual_det_loss(anchor, positives, negatives, margin, dimensions)

    return ms * 5.0 + res


def evil_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss = evil_triplet_loss(q_vec, pos_vecs, neg_vecs, m1)

    worst_pos = worst_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg), 1])
    worst_pos = tf.tile(tf.reshape(worst_pos, (-1, 1)), [1, int(num_neg)])
    m2 = tf.fill([int(batch), int(num_neg)], m2)

    second_loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(
        tf.add(m2, tf.subtract(worst_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, other_neg_copies), 2))),
        tf.zeros([int(batch), int(num_neg)])), 1))

    total_loss = trip_loss + second_loss

    return total_loss


def worst_pos_distance(query, pos_vecs):
    with tf.name_scope('best_pos_distance') as scope:
        num_pos = pos_vecs.get_shape()[1]
        query_copies = tf.tile(query, [1, int(num_pos), 1])  # shape num_pos x output_dim
        best_pos = tf.reduce_max(tf.reduce_sum(tf.squared_difference(pos_vecs, query_copies), 2), 1)
        return best_pos


def distance_loss(a_feature, pos_feature, squared_d_dists, d_max_squared, f_max_squared):
    scaled_d_dists, scaled_f_dists = _scale_distances(a_feature, pos_feature, squared_d_dists, d_max_squared,
                                                      f_max_squared)
    squared_diffs = tf.squared_difference(scaled_f_dists, scaled_d_dists)
    sum_of_squared_diffs = tf.reduce_mean(squared_diffs, 1)  # Mean over all positives
    return tf.reduce_mean(sum_of_squared_diffs, 0)  # Mean over batches


def huber_distance_loss(a_feature, pos_feature, squared_d_dists, d_max_squared, f_max_squared):
    scaled_d_dists, scaled_f_dists = _scale_distances(a_feature, pos_feature, squared_d_dists, d_max_squared,
                                                      f_max_squared)
    return tf.losses.huber_loss(scaled_d_dists, scaled_f_dists)


def distance_triplet_loss(a_feature, pos_features, neg_features, margin, lam, squared_d_dists, d_max_squared,
                          f_max_squared, triplet_loss_name='triplet_loss', distance_loss_name='huber_distance_loss'):
    """
    :param a_feature: Anchor
    :param pos_features: Positives
    :param neg_features: Negatives
    :param margin:
    :param lam: Scaling factor: loss = trip + lam*dist
    :param squared_d_dists: Squared distances from anchor to positives
    :param d_max_squared: Maximal squared distance
    :param f_max_squared: Maximal squared feature distance
    :param triplet_loss_name: triplet_loss or lazy_triplet_loss
    :param distance_loss_name: distance_loss or huber_distance_loss
    :return: loss
    """

    if 'huber' in distance_loss_name:
        return tf.add(getattr(pointnetvlad_cls, triplet_loss_name)
                      (a_feature, pos_features, neg_features, margin),
                      tf.multiply(lam, huber_distance_loss(a_feature, pos_features, squared_d_dists, d_max_squared,
                                                           f_max_squared)))
    else:
        return tf.add(getattr(pointnetvlad_cls, triplet_loss_name)
                      (a_feature, pos_features, neg_features, margin),
                      tf.multiply(lam, distance_loss(a_feature, pos_features, squared_d_dists, d_max_squared,
                                                     f_max_squared)))


def distance_quadruplet_loss(a_feature, pos_features, neg_features, other_neg, m1, m2, lam, squared_d_dists,
                             d_max_squared, f_max_squared, triplet_loss_name='triplet_loss',
                             distance_loss_name='huber_distance_loss'):
    """
    :param a_feature: Anchor
    :param pos_features: Positives
    :param neg_features: Negatives
    :param other_neg: Other negative
    :param m1:
    :param m2:
    :param lam: Scaling factor: loss = trip + lam*dist
    :param squared_d_dists: Squared distances from anchor to positives
    :param d_max_squared: Maximal squared distance
    :param f_max_squared: Maximal squared feature distance
    :param triplet_loss_name: triplet_loss or lazy_triplet_loss
    :param distance_loss_name: distance_loss or huber_distance_loss
    :return: loss
    """
    trip_loss = distance_triplet_loss(a_feature, pos_features, neg_features, m1, lam, squared_d_dists, d_max_squared,
                                      f_max_squared, triplet_loss_name, distance_loss_name)

    if 'huber' in distance_loss_name:
        best_pos = _best_huber_distance(a_feature, pos_features, squared_d_dists, d_max_squared, f_max_squared)
    else:
        best_pos = _best_distance(a_feature, pos_features, squared_d_dists, d_max_squared, f_max_squared)

    num_neg = neg_features.get_shape()[1]
    batch = a_feature.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m2 = tf.fill([int(batch), int(num_neg)], m2)
    f_max_copies = tf.fill([int(batch), int(num_neg)], f_max_squared)

    second_loss = tf.reduce_mean(tf.reduce_max(tf.maximum(
        tf.add(m2, tf.subtract(best_pos, tf.div(tf.reduce_sum(tf.squared_difference(neg_features, other_neg_copies), 2),
                                                f_max_copies))),
        tf.zeros([int(batch), int(num_neg)])), 1))

    total_loss = trip_loss + second_loss
    return total_loss


def neg_eigenvalue_loss(anchor, negatives):
    features = tf.concat([anchor, negatives], 1)
    min_eigenvalues = _min_eigenvalues(features)
    return tf.negative(tf.reduce_mean(min_eigenvalues))


# Min/Max eigenvalue loss
def ntuplet_evmm_loss(anchor, positives, negatives, margin):
    pos_features = tf.concat([anchor, positives], 1)
    neg_features = tf.concat([anchor, negatives], 1)

    max_neg_eigenvalues = _max_eigenvalues(neg_features)
    min_pos_eigenvalues = _min_eigenvalues(pos_features)

    batch = anchor.get_shape()[0]
    m = tf.fill([int(batch)], margin)
    losses = tf.maximum(tf.add(m, tf.subtract(min_pos_eigenvalues, max_neg_eigenvalues)), tf.zeros([int(batch)]))
    return tf.reduce_mean(losses)


# Sum of eigenvalue == trace loss
def ntuplet_trace_loss(anchor, positives, negatives, margin):
    pos_features = tf.concat([anchor, positives], 1)
    neg_features = tf.concat([anchor, negatives], 1)

    neg_trace = _trace(neg_features)
    pos_trace = _trace(pos_features)

    batch = anchor.get_shape()[0]
    m = tf.fill([int(batch)], margin)
    losses = tf.maximum(tf.add(m, tf.subtract(pos_trace, neg_trace)), tf.zeros([int(batch)]))
    return tf.reduce_mean(losses)


# Min/Max eigenvalue loss
def residual_det_loss(anchor, positives, negatives, margin, dimensions=10):
    num_pos = positives.get_shape()[1]
    num_neg = negatives.get_shape()[1]

    pos_features = tf.subtract(positives, tf.tile(anchor, [1, int(num_pos), 1]))
    neg_features = tf.subtract(negatives, tf.tile(anchor, [1, int(num_neg), 1]))

    pos_s = tf.slice(tf.linalg.svd(pos_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])
    neg_s = tf.slice(tf.linalg.svd(neg_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])

    losses = tf.add(tf.subtract(tf.reduce_prod(pos_s, axis=1), tf.reduce_prod(neg_s, axis=1)), margin)
    return tf.reduce_mean(losses, axis=0)  # Mean over batches


def swrd_loss(anchor, positives, negatives, pos_weights, neg_weights, margin, dimensions=10):
    num_pos = positives.get_shape()[1]
    num_neg = negatives.get_shape()[1]

    pos_features = tf.multiply(tf.subtract(positives, tf.tile(anchor, [1, int(num_pos), 1])), pos_weights)
    neg_features = tf.multiply(tf.subtract(negatives, tf.tile(anchor, [1, int(num_neg), 1])), neg_weights)

    pos_s = tf.slice(tf.linalg.svd(pos_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])
    neg_s = tf.slice(tf.linalg.svd(neg_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])

    losses = tf.add(tf.subtract(tf.reduce_prod(pos_s, axis=1), tf.reduce_prod(neg_s, axis=1)), margin)
    return tf.reduce_mean(losses, axis=0)  # Mean over batches


def wrd_loss(anchor, positives, negatives, pos_weights, neg_weights, margin, dimensions=10):
    num_pos = positives.get_shape()[1]
    num_neg = negatives.get_shape()[1]

    pos_features = tf.subtract(positives, tf.tile(anchor, [1, int(num_pos), 1]))
    neg_features = tf.subtract(negatives, tf.tile(anchor, [1, int(num_neg), 1]))

    all_features = tf.concat([pos_features, neg_features], 1)

    w_pos_features = tf.multiply(all_features, pos_weights)
    w_neg_features = tf.multiply(all_features, neg_weights)

    pos_s = tf.slice(tf.linalg.svd(w_pos_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])
    neg_s = tf.slice(tf.linalg.svd(w_neg_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])

    losses = tf.add(tf.subtract(tf.reduce_prod(pos_s, axis=1), tf.reduce_prod(neg_s, axis=1)), margin)
    return tf.reduce_mean(losses, axis=0)  # Mean over batches


def prodwrd_loss(anchor, positives, negatives, pos_weights, neg_weights, margin, dimensions=10, f_alpha_p=2.0,
                 f_alpha_n=50.0, f_lamb=1.0):

    num_pos = positives.get_shape()[1]
    num_neg = negatives.get_shape()[1]

    all_others = tf.concat([positives, negatives], 1)
    similarities = tf.matmul(anchor, all_others, transpose_a=False, transpose_b=True)

    all_residuals = tf.subtract(all_others, tf.tile(anchor, [1, int(num_pos) + int(num_neg), 1]))

    fw_pos = tf.transpose(tf.divide(1.0, (1.0 + tf.exp(f_alpha_p * (similarities - f_lamb)))),[0, 2, 1])
    fw_neg = tf.transpose(tf.divide(1.0, (1.0 + tf.exp(f_alpha_n * (f_lamb - similarities)))),[0, 2, 1])

    w_pos_features = tf.multiply(tf.multiply(all_residuals, pos_weights), fw_pos)
    w_neg_features = tf.multiply(tf.multiply(all_residuals, neg_weights), fw_neg)

    pos_s = tf.slice(tf.linalg.svd(w_pos_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])
    neg_s = tf.slice(tf.linalg.svd(w_neg_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])

    losses = tf.add(tf.subtract(tf.reduce_prod(pos_s, axis=1), tf.reduce_prod(neg_s, axis=1)), margin)
    return tf.reduce_mean(losses, axis=0)  # Mean over batches


def sumwrd_loss(anchor, positives, negatives, pos_weights, neg_weights, margin, dimensions=10, f_alpha_p=2.0,
                f_alpha_n=50.0, f_lamb=1.0):

    num_pos = positives.get_shape()[1]
    num_neg = negatives.get_shape()[1]

    all_others = tf.concat([positives, negatives], 1)
    similarities = tf.matmul(anchor, all_others, transpose_a=False, transpose_b=True)

    all_residuals = tf.subtract(all_others, tf.tile(anchor, [1, int(num_pos) + num_neg, 1]))

    fw_pos = tf.transpose(tf.divide(1.0, (1.0 + tf.exp(f_alpha_p * (similarities - f_lamb)))), [0, 2, 1])
    fw_neg = tf.transpose(tf.divide(1.0, (1.0 + tf.exp(f_alpha_n * (f_lamb - similarities)))), [0, 2, 1])

    w_pos_features = tf.multiply(all_residuals, pos_weights + fw_pos)
    w_neg_features = tf.multiply(all_residuals, neg_weights + fw_neg)

    pos_s = tf.slice(tf.linalg.svd(w_pos_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])
    neg_s = tf.slice(tf.linalg.svd(w_neg_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])

    losses = tf.add(tf.subtract(tf.reduce_prod(pos_s, axis=1), tf.reduce_prod(neg_s, axis=1)), margin)
    return tf.reduce_mean(losses, axis=0)  # Mean over batches




def incremental_s(X_in, s_old, v_old, m_old, seen):
    num_res = int(X_in.get_shape()[1])
    batch_size = int(X_in.get_shape()[0])
    mX = tf.reduce_mean(X_in, axis=1, keep_dims=True)
    mX_tiles = tf.tile(mX, [1, num_res, 1])

    print(m_old.get_shape())
    print(mX.get_shape())
    X_zero_mean = tf.subtract(X_in, mX_tiles)
    B = tf.concat([tf.matmul(tf.linalg.diag(s_old), v_old),
                   X_zero_mean,
                   tf.reshape(tf.multiply(
                       tf.sqrt(tf.truediv(tf.multiply(seen, num_res),
                                          tf.add(seen, num_res))), tf.subtract(mX, m_old)),
                       [batch_size, 1, -1])],
                  axis=1)

    return tf.linalg.svd(B, compute_uv=False)


# Min/Max eigenvalue loss
def incremental_residual_det_loss(anchor, positives, negatives, margin, s_old, v_old, m_old, seen, dimensions=10,
                                  scale=False):
    batches = int(anchor.get_shape()[0])

    s_old = tf.tile(tf.expand_dims(s_old, 0), [batches, 1])
    v_old = tf.tile(tf.expand_dims(v_old, 0), [batches, 1, 1])
    m_old = tf.tile(tf.expand_dims(m_old, 0), [batches, 1])
    m_old = tf.expand_dims(m_old, 1)

    num_pos = positives.get_shape()[1]
    num_neg = negatives.get_shape()[1]

    pos_features = tf.subtract(positives, tf.tile(anchor, [1, int(num_pos), 1]))
    neg_features = tf.subtract(negatives, tf.tile(anchor, [1, int(num_neg), 1]))

    residuals = tf.concat([pos_features, neg_features], 1)
    f_dim = int(residuals.get_shape()[2])

    incremental_pos = incremental_s(pos_features, s_old, v_old, m_old, seen)
    incremental_neg = incremental_s(neg_features, s_old, v_old, m_old, seen)

    num_s = int(incremental_pos.get_shape()[1])

    dimensions = tf.minimum(dimensions, num_s - 1)

    if scale:  # Necessary for large dim
        max_neg = tf.slice(incremental_neg, begin=[0, 0], size=[-1, 1])

        pos_s = tf.truediv(tf.slice(incremental_pos, begin=[0, 0], size=[-1, dimensions]),
                           tf.tile(max_neg, [1, dimensions]))
        neg_s = tf.truediv(tf.slice(incremental_neg, begin=[0, 0], size=[-1, dimensions]),
                           tf.tile(max_neg, [1, dimensions]))
    else:
        pos_s = tf.slice(incremental_pos, begin=[0, 0], size=[-1, dimensions])
        neg_s = tf.slice(incremental_neg, begin=[0, 0], size=[-1, dimensions])

    losses = tf.add(tf.subtract(tf.reduce_prod(pos_s, axis=1), tf.reduce_prod(neg_s, axis=1)), margin)
    return tf.reduce_mean(losses, axis=0), tf.reshape(residuals, [-1, f_dim])


# Min/Max eigenvalue loss
def incremental_residual_mm_loss(anchor, positives, negatives, margin, s_old, v_old, m_old, seen, dimensions=10,
                                 scale=False):
    batches = int(anchor.get_shape()[0])

    s_old = tf.tile(tf.expand_dims(s_old, 0), [batches, 1])
    v_old = tf.tile(tf.expand_dims(v_old, 0), [batches, 1, 1])
    m_old = tf.tile(tf.expand_dims(m_old, 0), [batches, 1])
    m_old = tf.expand_dims(m_old, 1)

    num_pos = positives.get_shape()[1]
    num_neg = negatives.get_shape()[1]

    pos_features = tf.subtract(positives, tf.tile(anchor, [1, int(num_pos), 1]))
    neg_features = tf.subtract(negatives, tf.tile(anchor, [1, int(num_neg), 1]))

    residuals = tf.concat([pos_features, neg_features], 1)
    f_dim = int(residuals.get_shape()[2])

    incremental_pos = incremental_s(pos_features, s_old, v_old, m_old, seen)
    incremental_neg = incremental_s(neg_features, s_old, v_old, m_old, seen)

    num_s = int(incremental_pos.get_shape()[1])

    dimensions = tf.minimum(dimensions, num_s - 1)

    if scale:
        max_neg = tf.slice(incremental_neg, begin=[0, 0], size=[-1, 1])

        pos_s = tf.truediv(tf.slice(incremental_pos, begin=[0, 0], size=[-1, dimensions]),
                           tf.tile(max_neg, [1, dimensions]))
        neg_s = tf.truediv(tf.slice(incremental_neg, begin=[0, 0], size=[-1, dimensions]),
                           tf.tile(max_neg, [1, dimensions]))
    else:
        pos_s = tf.slice(incremental_pos, begin=[0, 0], size=[-1, dimensions])
        neg_s = tf.slice(incremental_neg, begin=[0, 0], size=[-1, dimensions])

    losses = tf.add(tf.subtract(tf.reduce_max(pos_s, axis=1), tf.reduce_min(neg_s, axis=1)), margin)
    return tf.reduce_mean(losses, axis=0), tf.reshape(residuals, [-1, f_dim])


# Min/Max eigenvalue loss
def incremental_det_loss(anchor, positives, negatives, margin, s_old, v_old, m_old, seen, dimensions=10, scale=False):
    batches = int(anchor.get_shape()[0])

    s_old = tf.tile(tf.expand_dims(s_old, 0), [batches, 1])
    v_old = tf.tile(tf.expand_dims(v_old, 0), [batches, 1, 1])
    m_old = tf.tile(tf.expand_dims(m_old, 0), [batches, 1])
    m_old = tf.expand_dims(m_old, 1)

    pos_features = tf.concat([anchor, positives], 1)
    neg_features = tf.concat([anchor, negatives], 1)

    incremental_pos = incremental_s(pos_features, s_old, v_old, m_old, seen)
    incremental_neg = incremental_s(neg_features, s_old, v_old, m_old, seen)

    num_s = int(incremental_pos.get_shape()[1])

    dim = tf.minimum(dimensions, num_s - 1)

    if scale:
        max_neg = tf.slice(incremental_neg, begin=[0, 0], size=[-1, 1])

        pos_s = tf.truediv(tf.slice(incremental_pos, begin=[0, 0], size=[-1, dimensions]),
                           tf.tile(max_neg, [1, dimensions]))
        neg_s = tf.truediv(tf.slice(incremental_neg, begin=[0, 0], size=[-1, dimensions]),
                           tf.tile(max_neg, [1, dimensions]))
    else:
        pos_s = tf.slice(incremental_pos, begin=[0, 0], size=[-1, dimensions])
        neg_s = tf.slice(incremental_neg, begin=[0, 0], size=[-1, dimensions])

    losses = tf.add(tf.subtract(tf.reduce_prod(pos_s, axis=1), tf.reduce_prod(neg_s, axis=1)), margin)
    return tf.reduce_mean(losses, axis=0)  # Mean over batches


# Min/Max eigenvalue loss
def incremental_mm_loss(anchor, positives, negatives, margin, s_old, v_old, m_old, seen, dimensions=10, scale=False):
    batches = int(anchor.get_shape()[0])

    s_old = tf.tile(tf.expand_dims(s_old, 0), [batches, 1])
    v_old = tf.tile(tf.expand_dims(v_old, 0), [batches, 1, 1])
    m_old = tf.tile(tf.expand_dims(m_old, 0), [batches, 1])
    m_old = tf.expand_dims(m_old, 1)

    pos_features = tf.concat([anchor, positives], 1)
    neg_features = tf.concat([anchor, negatives], 1)

    incremental_pos = incremental_s(pos_features, s_old, v_old, m_old, seen)
    incremental_neg = incremental_s(neg_features, s_old, v_old, m_old, seen)

    num_s = int(incremental_pos.get_shape()[1])

    dim = tf.minimum(dimensions, num_s - 1)

    if scale:
        max_neg = tf.slice(incremental_neg, begin=[0, 0], size=[-1, 1])

        pos_s = tf.truediv(tf.slice(incremental_pos, begin=[0, 0], size=[-1, dimensions]),
                           tf.tile(max_neg, [1, dimensions]))
        neg_s = tf.truediv(tf.slice(incremental_neg, begin=[0, 0], size=[-1, dimensions]),
                           tf.tile(max_neg, [1, dimensions]))
    else:
        pos_s = tf.slice(incremental_pos, begin=[0, 0], size=[-1, dimensions])
        neg_s = tf.slice(incremental_neg, begin=[0, 0], size=[-1, dimensions])

    losses = tf.add(tf.subtract(tf.reduce_max(pos_s, axis=1), tf.reduce_min(neg_s, axis=1)), margin)
    return tf.reduce_mean(losses, axis=0)  # Mean over batches


# Sum of eigenvalue == trace loss
def residual_trace_loss(anchor, positives, negatives, margin, dimensions=10):
    num_pos = positives.get_shape()[1]
    num_neg = negatives.get_shape()[1]

    pos_features = tf.subtract(positives, tf.tile(anchor, [1, int(num_pos), 1]))
    neg_features = tf.subtract(negatives, tf.tile(anchor, [1, int(num_neg), 1]))

    pos_s = tf.slice(tf.linalg.svd(pos_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])
    neg_s = tf.slice(tf.linalg.svd(neg_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])

    losses = tf.add(tf.subtract(tf.reduce_sum(pos_s, axis=1), tf.reduce_sum(neg_s, axis=1)), margin)
    return tf.reduce_mean(losses, axis=0)  # Mean over batches


def pairwise_distance_loss(anchor, positives, pairwise_squared_d_dists, d_max_squared, f_max_squared,
                           distance_loss_name='distance_loss'):
    all_features = tf.concat([anchor, positives], 1)
    pairwise_squared_f_dists = _pairwise_squared_distances(all_features)

    # Scale distances
    d_max_copies = tf.fill(pairwise_squared_d_dists.get_shape(), d_max_squared)
    f_max_copies = tf.fill(pairwise_squared_f_dists.get_shape(), f_max_squared)

    scaled_d_dists = tf.div(pairwise_squared_d_dists, d_max_copies)
    scaled_f_dists = tf.div(pairwise_squared_f_dists, f_max_copies)

    if 'huber' in distance_loss_name:
        squared_diffs = tf.losses.huber_loss(scaled_f_dists, scaled_d_dists,
                                             reduction=tf.losses.Reduction.NONE)
    else:
        squared_diffs = tf.squared_difference(scaled_f_dists, scaled_d_dists)
    summed_diffs1 = tf.reduce_mean(squared_diffs, axis=2)
    summed_diffs2 = tf.reduce_mean(summed_diffs1, axis=1)
    return tf.reduce_mean(summed_diffs2, axis=0)


# Helper functions
def _features2eigenvalues(features):
    gram = tf.matmul(features, features, transpose_b=True)
    eig, _ = tf.linalg.eigh(gram)
    return eig


def _pairwise_squared_distances(features):
    num_batches = features.get_shape()[0]
    r = tf.einsum('aij,aij->ai', features, features)
    r = tf.reshape(r, [num_batches, -1, 1])
    batch_product = tf.einsum('aij,ajk->aik', features, tf.transpose(features, perm=[0, 2, 1]))
    return r - 2 * batch_product + tf.transpose(r, perm=[0, 2, 1])


def _best_distance(a_feature, pos_features, squared_d_dists, d_max_squared, f_max_squared):
    scaled_d_dists, scaled_f_dists = _scale_distances(a_feature, pos_features, squared_d_dists, d_max_squared,
                                                      f_max_squared)
    squared_diffs = tf.squared_difference(scaled_f_dists, scaled_d_dists)
    return tf.reduce_min(squared_diffs, 1)


def _best_huber_distance(a_feature, pos_features, squared_d_dists, d_max_squared, f_max_squared):
    scaled_d_dists, scaled_f_dists = _scale_distances(a_feature, pos_features, squared_d_dists, d_max_squared,
                                                      f_max_squared)
    squared_diffs = tf.losses.huber_loss(scaled_f_dists, scaled_d_dists, reduction=tf.losses.Reduction.NONE)
    return tf.reduce_min(squared_diffs, 1)


def _scale_distances(a_feature, pos_feature, squared_d_dists, d_max_squared, f_max_squared):
    num_pos = pos_feature.get_shape()[1]
    batch_size = a_feature.get_shape()[0]
    a_feature_copies = tf.tile(a_feature, [1, int(num_pos), 1])  # shape num_pos x output_dim
    squared_f_dists = tf.reduce_sum(tf.squared_difference(pos_feature, a_feature_copies), 2)

    d_max_copies = tf.fill([int(batch_size), int(num_pos)], d_max_squared)
    f_max_copies = tf.fill([int(batch_size), int(num_pos)], f_max_squared)

    scaled_d_dists = tf.div(squared_d_dists, d_max_copies)
    scaled_f_dists = tf.div(squared_f_dists, f_max_copies)

    return scaled_d_dists, scaled_f_dists


def _min_eigenvalues(features):
    return tf.reduce_min(_features2eigenvalues(features), axis=1)


def _max_eigenvalues(features):
    return tf.reduce_max(_features2eigenvalues(features), axis=1)


def _trace(features):
    gram = tf.matmul(features, features, transpose_b=True)
    return tf.linalg.trace(gram)


if __name__ == '__main__':
    # Testing _pairwise_squared_distances
    A = tf.constant([[[1.0, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [4, 3]]])
    B = tf.constant([[[1.0, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [4, 4]]])
    C = tf.reshape(B, [2, 3, 2])
    D = _pairwise_squared_distances(C)
    sess = tf.Session()
    print(sess.run(C))
    print(sess.run(D))

