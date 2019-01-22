from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles, rdFMCS, rdChemReactions, rdMolDescriptors
import itertools

import numpy as np

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
        member1_stripped_smiles = Chem.MolToSmiles(member1_stripped)
        member2_stripped_smiles = Chem.MolToSmiles(member2_stripped)

    # construct SMILES string from the two members:
        reaction = str(member1_stripped_smiles) + ">>" + str(member2_stripped_smiles)
        member1 = str(member1_stripped_smiles)
        member2 = str(member2_stripped_smiles)

    # combine all results (name of perturbation, reaction SMILES, ligand A Smiles and ligand B SMILES)
        result = [perturbation, reaction, member1, member2]
        perturbation_reactions.append(result)
     
    return perturbation_reactions



def build_deltaFP(reactions):
    PerturbationFingerprints = [[
    "Perturbation", 
    "Reaction_SMILES", 
    "ligandA_SMILES", 
    "ligandB_SMILES", 
    "Member_Similarity (Dice)", 
    "Perturbation Fingerprint (256 bits)"]]

    for reaction_members in reactions:
        pert = str(reaction_members[0])
    # take mol object from each member
        member1 = Chem.MolFromSmiles(reaction_members[2])
        member2 = Chem.MolFromSmiles(reaction_members[3])

    # create bitstring of 256 bits for each member. Max values between 1 and 3
        FP1 = (rdMolDescriptors.GetHashedAtomPairFingerprint(member1, 256, 1, 3))
        FP2 = (rdMolDescriptors.GetHashedAtomPairFingerprint(member2, 256, 1, 3))
        similarity = DataStructs.DiceSimilarity(FP1, FP2)

    # subtract and return reaction FP (=deltaFP) as list
        deltaFP = np.array(list(FP2)) - np.array(list(FP1))
#        print("Perturbation FP for " + pert +" is:")
#        print(deltaFP)

    # join all the data together into one list and append to output:
        result = reaction_members + ([str(similarity)]) + deltaFP.tolist()
        PerturbationFingerprints.append(result)
        
        print(str(reaction_members[0]) + ":")
        print(reaction_members[1])
        print("##########")
    return PerturbationFingerprints







perturbations_all_paths = read_morph_file(MorphFilePath)

reactions = build_reactions(perturbations_all_paths)

build_deltaFP(reactions)
