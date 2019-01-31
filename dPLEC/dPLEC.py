import oddt
from oddt import fingerprints

import numpy as np
np.set_printoptions(edgeitems=10)

import itertools
import csv
import os



MorphFilePath = '../fesetup/morph.in'
PathToProtein = '../fesetup/protein/BACE/protein.pdb'

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
        print("Amount of perturbations:",len(perturbation_list))
        print("#####################################")
    
    # replace ligand names with paths to their respective pdb files:
    perturbations_paths = []
    perturbation_names = []
    for morph_pair in perturbation_list:
        member1_path = "../fesetup/poses/" + str(morph_pair[0]) + "/ligand.pdb"
        member2_path = "../fesetup/poses/" + str(morph_pair[1]) + "/ligand.pdb"
        perturbations_paths.append([member1_path, member2_path])

    return perturbations_paths


def pdbPaths_to_MOL2s(perturbations_paths):
    newpaths = []

    # convert pdb files to mol2 because then rdkit can read in bond orders:
    print("Converting PDB ligands to MOL2..")
    os.system("for f in ../fesetup/poses/*/ligand.pdb; do \
        mol2string=$(echo $f | sed 's/.pdb/.mol2/g'); \
        obabel -i pdb $f -O $mol2string; \
        done >/dev/null 2>&1")

    # load mol2 paths, compile to list:
    for pert in perturbations_paths:

        member1_molpath = pert[0].replace(".pdb", ".mol2")
        member1_mol = next(oddt.toolkit.readfile("mol2", member1_molpath))

        member2_molpath = pert[1].replace(".pdb", ".mol2")
        member2_mol = next(oddt.toolkit.readfile("mol2", member2_molpath))
        
        newpaths.append([member1_mol, member2_mol])
    return newpaths

def calcFPs(perturbation_mols_list, path_to_protein, fpsize):	
	# Generate protein ODDT mol object:
    try:
        protein = next(oddt.toolkit.readfile("pdb", path_to_protein))
    except:
        print("Failed to generate protein object")
        return

    # Tell ODDT that the protein variable is a protein structure:
    protein.protein = True  

    # Make PLEC function:
    def PLEC_FP(ligand, protein):
        PLEC = fingerprints.PLEC(ligand, protein, 
        sparse=False,                           # equal bitsize for all processed ligands
        ignore_hoh=False,                       # incorporate H20 interactions in FP
        depth_ligand=1,                         # recommended ligand ECFP depth
        depth_protein=5,                        # recommended protein ECFP depth
        distance_cutoff=3.5,                    # maximum distance for atomic interaction
        size=fpsize)                            # 4096, 16384, 32768 or 65536 bits
        return np.array(PLEC, dtype="int64")    # change dtype from uint8 to int64 to allow subtraction


    # Call PLEC with set parameters for each ligand pair;
    # compute dPLEC by subtracting ligand A PLEC from ligand B PLEC:
    print("Computing dPLEC per perturbation..")
    dPLECs = []
    for ligand_pair in perturbation_mols_list:
        
        FP_Pair = [ PLEC_FP(ligand, protein) for ligand in ligand_pair ]
        deltaPLEC = FP_Pair[1] - FP_Pair[0]
        dPLECs.append(deltaPLEC.tolist())

    return dPLECs




def MergeWriteNamesAndPLECs(pdbs, deltaPLECs, fpsize):

    # Isolate perturbation name from source paths:
    perturbation_names = []
    for pair in pdbs:
        pert = [ path.replace("../fesetup/poses/","").replace("/ligand.pdb","") for path in pair ]
        pert_name = str(pert[0]+">"+str(pert[1]))
        perturbation_names.append([pert_name])

    # Pair perturbation names and corresponding PLECs:
    zipped_names_and_PLECs = [ list(merged) for merged in zip(perturbation_names, deltaPLECs)]
    flattened_data = [ item[0] + item[1] for item in zipped_names_and_PLECs]
    

    # Write to file:
    if not os.path.exists("./dPLECs_output"):
        os.makedirs("./dPLECs_output")
    print("Writing to \'./dPLECs_output/perts_dPLECs.csv\'")

    with open('./dPLECs_output/perts_dPLECs.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        PLEC_index = np.arange(0, fpsize).tolist()
        colnames = ["plec" + str(index) for index in PLEC_index]
        columns = ["Perturbation"] + colnames
        
        writer.writerow(columns)
        for row in flattened_data:
            writer.writerow(row)


    #print(flattened_data)


pdbs = read_morph_file(MorphFilePath)
mol_paths = pdbPaths_to_MOL2s(pdbs)
deltaPLECs = calcFPs(mol_paths, PathToProtein, 16384)

MergeWriteNamesAndPLECs(pdbs, deltaPLECs, 16384)