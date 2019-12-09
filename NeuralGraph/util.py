import torch as T
import numpy as np
dev = T.device('cuda') if T.cuda.is_available() else T.device('cpu')


def temporal_padding(x, paddings=(1, 0), pad_value=0):
    """Pad the middle dimension of a 3D tensor
        with `padding[0]` values left and `padding[1]` values right.
        Modified from keras.backend.temporal_padding
        https://github.com/fchollet/keras/blob/3bf913d/keras/backend/theano_backend.py#L590
    """
    if not isinstance(paddings, (tuple, list, np.ndarray)):
        paddings = (paddings, paddings)
    output = T.zeros(x.size(0), x.size(1) + sum(paddings), x.size(2)).to(dev)
    output[:, :paddings[0], :] = pad_value
    output[:, paddings[1]:, :] = pad_value
    output[:, paddings[0]: paddings[0]+x.size(1), :] = x
    return output


def lookup_neighbors(atoms, edges, maskvalue=0, include_self=False):
    """ Looks up the features of an all atoms neighbours, for a batch of molecules.
        # Arguments:
            atoms (Tensor): of size (batch_n, max_atoms, num_atom_features)
            edges (Tensor): of size (batch_n, max_atoms, max_degree) with neighbour
                indices and -1 as padding value
            maskvalue (numerical): the masking value that should be used for empty atoms
                or atoms that have no neighbours (does not affect the input maskvalue
                which should always be -1!)
            include_self (bool): if True, the feature vector of each atom will be added
                to the list feature vectors of its neighbours
        # Returns:
            neigbour_features (Tensor): of size (batch_n, max_atoms(+1), max_degree,
                num_atom_features) depending on the value of include_self
    """
    # The lookup masking trick: We add 1 to all indices, converting the
    #   masking value of -1 to a valid 0 index.
    masked_edges = edges + 1
    # We then add a padding vector at index 0 by padding to the left of the
    #   lookup matrix with the value that the new mask should get
    masked_atoms = temporal_padding(atoms, (1, 0), pad_value=maskvalue)

    # Import dimensions
    batch_n, lookup_size, n_atom_features = masked_atoms.size()
    _, max_atoms, max_degree = masked_edges.size()

    expanded_atoms = masked_atoms.unsqueeze(2).expand(batch_n, lookup_size, max_degree, n_atom_features)
    expanded_edges = masked_edges.unsqueeze(3).expand(batch_n, max_atoms, max_degree, n_atom_features)
    output = T.gather(expanded_atoms, 1, expanded_edges)

    if include_self:
        return T.cat([atoms.view(batch_n, max_atoms, 1, n_atom_features), output], dim=2)
    return output
