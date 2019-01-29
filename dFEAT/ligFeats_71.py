import pandas as pd 
from mordred import Calculator, descriptors
from rdkit import ML, Chem

import itertools
import csv
import os

# open user-specified descriptors and create list:
# consult http://mordred-descriptor.github.io/documentation/master/descriptors.html
descriptors_raw = open("./descriptors/used_descriptors.txt", "r")
descriptors_raw_list = [line.split("\n") for line in descriptors_raw.readlines()]
descriptors_list = [desc[0] for desc in descriptors_raw_list]
print("Amount of descriptors: " + str(len(descriptors_list)))

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

    # load mol2 files, return nested list with molecule object pairs
    for pert in perturbations_paths:

        member1_molfile = pert[0].replace(".pdb", ".mol2")
        member1 = Chem.MolFromMol2Block(open(member1_molfile,'r').read())

        member2_molfile = pert[1].replace(".pdb", ".mol2")
        member2 = Chem.MolFromMol2Block(open(member2_molfile,'r').read())
        newpaths.append([member1, member2])
    return newpaths


def ligFeaturesFromMols(mol_files, pert_paths):
    
    # set up feature calculator, run per perturbation pair and calculate the feature difference 
    # i.e. subtract each member2 value from each member1 value:
    print("Generating deltaFeatures (~2 perturbations p/s)..")
    calc = Calculator(descriptors, ignore_3D=False)

    subtraction_values = []
    for pert in mol_files:
        featured_members = calc.pandas(pert)
        featured_members_picked = featured_members[descriptors_list]
        
        featured_diff = featured_members_picked.diff(periods=1)
        subtraction_values.append(featured_diff.iloc[[1]])

    deltaFeatures = pd.concat(subtraction_values)

    # regenerate perturbation names:
    pert_names = []
    for pert in pert_paths:
        member1 = pert[0].replace("../fesetup/poses/", "").replace("/ligand.pdb","")
        member2 = pert[1].replace("../fesetup/poses/", "").replace("/ligand.pdb","")
        pert_names.append(str(member1) + ">" + str(member2))
    
    # gather data, merge with perturbation names and output as CSV
    results_csv = [["Perturbation"] + deltaFeatures.columns.tolist()]
    deltaFeatures = deltaFeatures.values.tolist()

    # merge perturbation names with the corresponding deltaFeature data:
    for i in range(len(list(pert_names))):
        results_csv.append([pert_names[i]] + deltaFeatures[i])

    # write to csv file 
    if not os.path.exists("./dFeatures_output"):
        os.makedirs("./dFeatures_output")

    with open('./dFeatures_output/deltaFeatures.csv', 'w') as csvfile:
        for row in results_csv:
            writer = csv.writer(csvfile)
            writer.writerow(row)
    print("Succes, wrote file to './dFeatures_output/deltaFeatures.csv'")

    return


perturbation_paths = read_morph_file(MorphFilePath)
mol2_files = pdbPaths_to_MOL2s(perturbation_paths)
ligFeaturesFromMols(mol2_files, perturbation_paths)