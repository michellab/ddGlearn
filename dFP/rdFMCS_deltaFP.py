from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles, rdFMCS, rdChemReactions, rdMolDescriptors

import itertools
import numpy as np
import os
import csv

#####################


MorphFilePath = '../fesetup/morph.in'


def read_morph_file(MorphFilePath):
    # read in morphfile:
    with open(MorphFilePath, 'rt') as morph_file:

    # read morph_pairs for cleaning
        block = list(itertools.takewhile(lambda x: "[protein]" not in x,
            itertools.dropwhile(lambda x: "morph_pairs" not in x, morph_file)))

        morph_list = [w.replace("\n", "").replace("\t","").replace(",", ", ") for w in block]
        morph_pairs = "".join(morph_list)
        
    # clean data and return as nested list:
        try:
            first_cleaned = (morph_pairs.replace("morph_pairs","").replace("=","").replace(",","\n"))
        except:
            print("Error in reading morph file, check if the line \"morph_pairs = ...\" is ordered vertically. Exiting..")
            return
        second_cleaned = (first_cleaned.replace(" ", "").replace(">",", "))
        molecule_pairs = second_cleaned.split("\n")
        perturbation_list = []
        for pert in molecule_pairs:
            if len(pert) != 0: 
                perturbation_list.append(pert.split(", ", 1))
        print("Total amount of perturbations is: ",len(perturbation_list))
        print("#####################################")
    
    # replace ligand names with paths to their respective pdb files:
    perturbations_paths = []
    for morph_pair in perturbation_list:
        member1_path = "../fesetup/poses/" + str(morph_pair[0]) + "/ligand.pdb"
        member2_path = "../fesetup/poses/" + str(morph_pair[1]) + "/ligand.pdb"
        perturbations_paths.append([member1_path, member2_path])

    return perturbations_paths


def DeleteSubstructs_unique(mol, submol):
    # this function is called in the rare case that the MCS substructure fits 'twice' in 
    # the larger perturbation member, deleting the whole structure resulting in a .. >> ..
    # this function defines a list of possible matches and takes only the first:
    matches = mol.GetSubstructMatches(submol)
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


def build_reactions(perturbations_all_paths):
    # loop over each perturbation in the list and load the pdb files:
    perturbation_reactions = []
    for perturbation_pair_path in perturbations_all_paths:

    # regenerate the perturbation (A>B):
        ligA = perturbation_pair_path[0].replace("../fesetup/poses/", "").replace("/ligand.pdb","")
        ligB = perturbation_pair_path[1].replace("../fesetup/poses/", "").replace("/ligand.pdb","")
        perturbation = str(ligA) + ">" + str(ligB)

    # read in PDB files:
        perturbation_pair = []
        member1_pdb_file = open(perturbation_pair_path[0], 'r').read()
        member2_pdb_file = open(perturbation_pair_path[1], 'r').read()  

        perturbation_pair.append(rdmolfiles.MolFromPDBBlock(member1_pdb_file))
        perturbation_pair.append(rdmolfiles.MolFromPDBBlock(member2_pdb_file))


    # generate MCS (taking into account substitutions in ring structures)
        print("Generating MCS for perturbation " + str(perturbation) + "..")
        MCS_object = rdFMCS.FindMCS(perturbation_pair, completeRingsOnly=True)
        MCS_SMARTS = Chem.MolFromSmarts(MCS_object.smartsString)

        if MCS_SMARTS == None:
            print("Could not generate MCS pattern")
            return

    # use SMARTS pattern to isolate unique patterns in each pair member
    # if multiple unique patterns exist in one molecule they are written as:
    # pattern1.pattern2 ('.' signifies a non-bonded connection)
        member1 = perturbation_pair[0]
        member2 = perturbation_pair[1]
        member1_stripped = AllChem.DeleteSubstructs(member1, MCS_SMARTS)
        member2_stripped = AllChem.DeleteSubstructs(member2, MCS_SMARTS)
        member1_stripped_smiles = Chem.MolToSmiles(member1_stripped, allHsExplicit=True)
        member2_stripped_smiles = Chem.MolToSmiles(member2_stripped, allHsExplicit=True)

    # when regular method creates a .. >> .., call alternative function:
        if len(member1_stripped_smiles) == 0 and len(member2_stripped_smiles) == 0:
            member1_stripped = DeleteSubstructs_unique(member1, MCS_SMARTS)
            member2_stripped = DeleteSubstructs_unique(member2, MCS_SMARTS)
            member1_stripped_smiles = Chem.MolToSmiles(member1_stripped[0])
            member2_stripped_smiles = Chem.MolToSmiles(member2_stripped[0])

    # if either member turns out empty, place a hydrogen for clarity (doesn't influence bits):            
        if len(member1_stripped_smiles) == 0:
            member1_stripped_smiles = "[H]"
        if len(member2_stripped_smiles) == 0:
            member2_stripped_smiles = "[H]"
    
    # if either member contains only a CH4, make it C-CH4 so the AP FP activates a C-C bond:
        if member1_stripped_smiles == "[CH4]":
            member1_stripped_smiles = "[C][CH4]"
        if member2_stripped_smiles == "[CH4]":
            member2_stripped_smiles = "[C][CH4]"
    # construct SMILES string from the two members (stripped and full):
        reaction = str(member1_stripped_smiles) + ">>" + str(member2_stripped_smiles)
        member1 = str(member1_stripped_smiles)
        member2 = str(member2_stripped_smiles)

        member1_fullsmiles = Chem.MolToSmiles(perturbation_pair[0])
        member2_fullsmiles = Chem.MolToSmiles(perturbation_pair[1])

    # combine all results (SMILES):
        result = [perturbation, reaction, member1_fullsmiles, member2_fullsmiles]

        perturbation_reactions.append(result)
     
    return perturbation_reactions



def build_deltaFP(reactions):
    print("Building FPs and writing to CSV..")
    FP_column = np.arange(1, 257).tolist()
    FP_column = [str(item) for item in FP_column]

    PerturbationFingerprints = [
    "Perturbation", 
    "Reaction_SMILES", 
    "fullmember1",
    "fullmember2",
    "Member_Similarity (Dice)", 
    ]
    PerturbationFingerprints = [PerturbationFingerprints + FP_column]
    for reaction_members in reactions:
        pert = str(reaction_members[0])
    # deconstruct reaction smiles back into members:    
        head, sep, tail = reaction_members[1].partition(">>")

    # take mol object from each member, retain hydrogens and override valency discrepancies
        member1 = Chem.MolFromSmiles(head, sanitize=False)
        member2 = Chem.MolFromSmiles(tail, sanitize=False)
        member1.UpdatePropertyCache(strict=False)
        member2.UpdatePropertyCache(strict=False)

     # create bitstring of 256 bits for each member. 
        FP1 = (rdMolDescriptors.GetHashedAtomPairFingerprint(member1, 256))
        FP2 = (rdMolDescriptors.GetHashedAtomPairFingerprint(member2, 256))
        similarity = DataStructs.DiceSimilarity(FP1, FP2)

     # subtract and return reaction FP (=deltaFP) as list
        deltaFP = np.array(list(FP2)) - np.array(list(FP1))
#        print("Perturbation FP for " + pert +" (" + str(reaction_members[1]) + ") is:")
#        print(deltaFP)
        
     # join all the data together into one list and append to output:
        result = reaction_members + ([str(similarity)]) + deltaFP.tolist()
        PerturbationFingerprints.append(result)
        
#        print("##########################################################################")
    return PerturbationFingerprints



perturbations_all_paths = read_morph_file(MorphFilePath)

reactions = build_reactions(perturbations_all_paths)

results_with_FPs = build_deltaFP(reactions)


if not os.path.exists("./dFP_output"):
    os.makedirs("./dFP_output")


with open('./dFP_output/perts_APFPs.csv', 'w') as csvfile:
    for row in results_with_FPs:
        writer = csv.writer(csvfile)
        writer.writerow(row)
