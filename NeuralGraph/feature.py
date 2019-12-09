import numpy as np
from rdkit import Chem


def one_of_k_encoding(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return one_of_k_encoding(atom.GetSymbol(),
                             ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Unknown']) \
           + one_of_k_encoding(atom.GetDegree(), list(range(7))) \
           + one_of_k_encoding(atom.GetTotalNumHs(), list(range(6))) \
           + one_of_k_encoding(atom.GetImplicitValence(), list(range(7))) \
           + one_of_k_encoding(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                SP3D, Chem.rdchem.HybridizationType.SP3D2]) \
           + [atom.GetIsAromatic()]


def atom_position(atom, conformer):
    return conformer.GetPositions()[atom.GetIndex()]


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()], dtype=np.float)


def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))


def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))
