try:
    from cresset import flare
except ImportError:
    flare=None
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles, rdFMCS
from rdkit.Chem.Draw import IPythonConsole

def DeleteSubstructs2(mol, submol):
    matches = mol.GetSubstructMatches(submol)
    print(mol)
    res = []
    for match in matches:
        match = [m for m in match if mol.GetAtomWithIdx(m).GetAtomicNum() > 1]
        exp_hs_to_add = []
        indices_to_remove = set()
        bonds_to_remove = set()
        mol_copy = Chem.Mol(mol)
        for b in mol_copy.GetBonds():
            is_ba_in_match = (b.GetBeginAtomIdx() in match)
            is_ea_in_match = (b.GetEndAtomIdx() in match)
            if (is_ba_in_match or is_ea_in_match):
                bonds_to_remove.add((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            if ((b.GetBeginAtom().GetAtomicNum() == 1 and is_ea_in_match)
                or (b.GetEndAtom().GetAtomicNum() == 1 and is_ba_in_match)):
                indices_to_remove.add(b.GetBeginAtomIdx() if is_ea_in_match
                                      else b.GetEndAtomIdx())
                continue
            if ((is_ba_in_match and (not is_ea_in_match))
                or (is_ea_in_match and (not is_ba_in_match))):
                if (is_ba_in_match):
                    a = b.GetEndAtom() if is_ba_in_match else b.GetBeginAtom()
                try:
                    exp_h_add = a.GetIntProp('__exp_h_add')
                except KeyError:
                    exp_h_add = 0
                exp_h_add += 1
                a.SetIntProp('__exp_h_add', exp_h_add)
        indices_to_remove_sorted = sorted(indices_to_remove.union(match),
                                          reverse=True)
        rwmol = Chem.RWMol(mol_copy)
        [rwmol.RemoveBond(ba, ea) for (ba, ea) in bonds_to_remove]
        [rwmol.RemoveAtom(i) for i in indices_to_remove_sorted]
        for a in rwmol.GetAtoms():
            try:
                exp_h_add = a.GetIntProp('__exp_h_add')
            except KeyError:
                continue
            a.SetNumExplicitHs(a.GetNumExplicitHs() + exp_h_add)
        mol_copy = Chem.AddHs(rwmol,
                              addCoords=(mol.GetNumConformers() > 0),
                              explicitOnly=True)
        Chem.SanitizeMol(mol_copy)
        mol_copy.ClearComputedProps()
        mol_copy.UpdatePropertyCache()
        res.append(mol_copy)
    return res

    # load molecules:
lig02_pdb = open("17.pdb", "r").read()
lig12_pdb = open("18.pdb", "r").read()

if flare:
    ligands = flare.main_window().project.ligands
    lig02 = ligands.extend(flare.read_string(lig02_pdb, "pdb"))[-1]
    lig12 = ligands.extend(flare.read_string(lig12_pdb, "pdb"))[-1]
    lig02_mol = lig02.to_rdmol()
    lig12_mol = lig12.to_rdmol()
else:
    lig02_mol = rdmolfiles.MolFromPDBBlock(lig02_pdb)
    lig12_mol = rdmolfiles.MolFromPDBBlock(lig12_pdb)

# make list of molecules to map the MCS to:
perturbation_pair = []
perturbation_pair.append(lig02_mol)
perturbation_pair.append(lig12_mol)

MCS_object = rdFMCS.FindMCS(perturbation_pair, completeRingsOnly=True)

MCS_SMARTS = Chem.MolFromSmarts(MCS_object.smartsString)


# remove MCS from each molecule:
lig02_stripped = AllChem.DeleteSubstructs(lig02_mol, MCS_SMARTS)
lig12_stripped = AllChem.DeleteSubstructs(lig12_mol, MCS_SMARTS)

# print SMILES of each stripped molecule:
print("lig02: " + str([Chem.MolToSmiles(m) for m in lig02_stripped]))
print("lig12: " + str([Chem.MolToSmiles(m) for m in lig12_stripped]))