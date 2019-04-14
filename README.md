# neural_graph_fingerprint
PyTorch-based Neural Graph Fingerprint for Organic Molecule Representations

# Keras Neural Graph Fingerprint

This repository is an implementation of [Convolutional Networks on Graphs for Learning Molecular Fingerprints][NGF-paper] in Keras.

It includes a preprocessing function to convert molecules in smiles representation
into molecule tensors.

## Related work

There are several implementations of this paper publicly available:
 - by [HIPS][1] using autograd
 - by [debbiemarkslab][2] using theano
 - by [GUR9000] [3] using keras
 - by [ericmjl][4] using autograd
 - by [DeepChem][5] using tensorflow
 - by [keiserlab][6] using Keras

The closest implementation is the implementation by GUR9000 and keiserlab in Keras. However this
repository represents moleculs in a fundamentally different way. The consequences
are described in the sections below.

## Molecule Representation

### Atom, bond and edge tensors
This codebase uses tensor matrices to represent molecules. Each molecule is
described by a combination of the following three tensors:

   - **atom matrix**, size: `(max_atoms, num_atom_features)`
   	 This matrix defines the atom features.

     Each column in the atom matrix represents the feature vector for the atom at
     the index of that column.

   - **edge matrix**, size: `(max_atoms, max_degree)`
     This matrix defines the connectivity between atoms.

     Each column in the edge matrix represent the neighbours of an atom. The
     neighbours are encoded by an integer representing the index of their feature
     vector in the atom matrix.

     As atoms can have a variable number of neighbours, not all rows will have a
     neighbour index defined. These entries are filled with the masking value of
     `-1`. (This explicit edge matrix masking value is important for the layers
     to work)

   - **bond tensor** size: `(max_atoms, max_degree, num_bond_features)`
   	 This matrix defines the atom features.

   	 The first two dimensions of this tensor represent the bonds defined in the
   	 edge tensor. The column in the bond tensor at the position of the bond index
   	 in the edge tensor defines the features of that bond.

   	 Bonds that are unused are masked with 0 vectors.


### Batch representations

 This codes deals with molecules in batches. An extra dimension is added to all
 of the three tensors at the first index. Their respective sizes become:

 - **atom matrix**, size: `(num_molecules, max_atoms, num_atom_features)`
 - **edge matrix**, size: `(num_molecules, max_atoms, max_degree)`
 - **bond tensor** size: `(num_molecules, max_atoms, max_degree, num_bond_features)`

As molecules have different numbers of atoms, max_atoms needs to be defined for
the entire dataset. Unused atom columns are masked by 0 vectors.

### Strong and weak points
The obvious downside of this representation is that there is a lot of masking,
resulting in a waste of computation power.

The alternative is to represent the entire dataset as a bag of atoms as in the
authors [original implementation](https://github.com/HIPS/neural-fingerprint). For
larger datasets, this is infeasable. In [keiserlab's implementation] (https://github.com/keiserlab/keras-neural-graph-fingerprint)
the same approach is used, but each batch is pre-calculated as a bag of atoms.
The downside of this is that each epoch uses the exact same composition of batches,
decreasing the stochasticity. Furthermore, Keras recognises the variability in batch-
size and will not run. In his implementation GUR9000 included a modified version
of Keras to correct for this.

The tensor representation used in this repository does not have these downsides,
and allows for many modificiations of Duvenauds algorithm (there is a lot to explore).


## Dependencies
- [**RDKit**](http://www.rdkit.org/) This dependency is nescecairy to convert molecules into tensor
representatins, once this step is conducted, the new data can be stored, and RDkit
is no longer a dependency.
- [**PyTorch**](https://PyTorch.org/) Requires PyTorch 1.0 for building, training and evaluating the models.
- [**NumPy**](http://www.numpy.org/)

## Acknowledgements
- Implementation is based on [Duvenaud et al., 2015][NGF-paper].
- Feature extraction scripts were implemented from [the original implementation][1]
- Data preprocessing scripts were rewritten from [keiserlab][3]
- Graphpool layer adopted from [Han, et al., 2016][DeepChem-paper]

[NGF-paper]: https://arxiv.org/abs/1509.09292
[DeepChem-paper]:https://arxiv.org/abs/1611.03199
[keiserlab]: //http://www.keiserlab.org/
[1]: https://github.com/HIPS/neural-fingerprint
[2]: https://github.com/debbiemarkslab/neural-fingerprint-theano
[3]: https://github.com/GUR9000/KerasNeuralFingerprint
[4]: https://github.com/ericmjl/graph-fingerprint
[5]: https://github.com/deepchem/deepchem
[6]: https://github.com/keiserlab/keras-neural-graph-fingerprint
