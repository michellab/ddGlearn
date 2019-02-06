import pandas as pd
import csv
import os
import numpy as np
import itertools


pFP_path = '../pFP/dFP_output/perts_APFPs.csv'
dFEAT_path = '../dFEAT/dFeatures_output/deltaFeatures.csv'
dPLEC_path = '../dPLEC/dPLECs_output/perts_dPLECs.csv'
experi_path = '../ddGlearner/input/experimental/experimental_dGs.csv'
summary_path = "summaries/summary.csv"
dataset_path = "datasets_compiled/dataset_123.csv"
MorphFilePath = '../fesetup/morph.in'
offsets_path  = 'ddG_offset_compiled/perts_ddG_offsets.csv'

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
                pert_pair = pert.split(", ", 1)
                perturbation_list.append([pert_pair[0] + ">" + pert_pair[1]])
        print("Amount of perturbations:",len(perturbation_list))
        print("#####################################")
    
    return perturbation_list


def load_ddGs_FEP(summary_path):
	with open(summary_path) as file:
		reader = csv.reader(file)

		ddGs_FEP = []
		excluded_ddGs = []
		for row in reader:
	# per line in file, take only lines starting with a ligand name:
			if not row[0].startswith("#"):
				
	# exclude nonsense (i.e. failed) perturbations:
				try:
					if float(row[2]) <= 50 and float(row[3]) <= 1:
						pert_string = row[0] + ">" + row[1]
						pert_value = float(row[2])
						ddGs_FEP.append([pert_string, pert_value])
					else:
						excluded_ddGs.append(row)
				except:
					print("Something is wrong with the formatting of the summary.csv")
					return
		print("Excluded " + str(len(excluded_ddGs)) + " nonsense predictions.")

		return ddGs_FEP


def load_ddGs_EXP(experi_path, perturbation_list):
	## experimental data extraction ##
	# open and process experimental dG_exps as dict:
	with open(experi_path, "r") as csvfile:
		experi = []
		reader = csv.reader(csvfile)
		for row in reader:
			experi.append(row)
	experi_dict = { k[0]:float(k[1]) for k in experi }

	# prepare nested list to map experimental values to:
	perts = perturbation_list
	ligpairs = [ pert[0].split(">") for pert in perts ]


	# map experimental values to perturbations
	ddGs_EXP = []
	for pair in ligpairs:
		ligAdG_exp = experi_dict.get(pair[0])
		ligBdG_exp = experi_dict.get(pair[1])
	# compute experimental dG_exp (if a member is not in experimental 
	# this is assumed to be a fictive intermediate):
		try:
			dG_exp = float(ligBdG_exp) - float(ligAdG_exp)
		except TypeError:
			dG_exp = "fictive"
		pert = pair[0] + ">" + pair[1]
		try:
			ddGs_EXP.append([pert, round(float(dG_exp), 4)])
		except ValueError:
			ddGs_EXP.append([pert, "fictive"])
	return ddGs_EXP


def compute_ddG_offsets(ddGs_FEP, ddGs_EXP):
	ddG_offsets = []
	excluded_perts = []

	# Match perturbations between FEP and EXP:
	for prediction in ddGs_FEP:
		for experi in ddGs_EXP:
			if prediction[0] == experi[0]:
				try:
	# Compute ddG offset:
					offset = prediction[1] - experi[1]
					ddG_offsets.append([prediction[0], round(offset, 3)]) 
	# Exclude perturbations where the experimental ddG is "fictive":
				except TypeError:
					excluded_perts.append(experi)
	print("Excluded " + str(len(excluded_perts)) + " perturbations for containing fictive ligands (i.e. intermediates).")
	print("Computed " + str(len(ddG_offsets)) + " ddG offsets")
	# write to file:
	if not os.path.exists("./ddG_offset_compiled"):
		os.makedirs("./ddG_offset_compiled")

	with open('./ddG_offset_compiled/perts_ddG_offsets.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["Perturbation", "ddG_offset"])
		for row in ddG_offsets:
			writer.writerow(row)

	print("Wrote offsets file to \'./ddG_offset_compiled/perts_ddG_offsets.csv\'")
	print("#####################################")
	return ddG_offsets



perturbation_list = read_morph_file(MorphFilePath)
ddGs_FEP = load_ddGs_FEP(summary_path)

ddGs_EXP = load_ddGs_EXP(experi_path, perturbation_list)

compute_ddG_offsets(ddGs_FEP, ddGs_EXP)

def build_dataset(pFP_path, dFEAT_path, dPLEC_path, offsets_path):
	# Read files and store to DFs:
	pFP = pd.read_csv(pFP_path, index_col=0)
	pFP_APFPs = pFP.drop(["Reaction_SMILES",		# remove unneeded columns
		"fullmember1", 
		"fullmember2", 
		"Member_Similarity (Dice)"], axis=1)
	dFEAT = pd.read_csv(dFEAT_path, index_col=0)
	dPLEC = pd.read_csv(dPLEC_path, index_col=0)

	# Merge DFs pairwise by index:
	dataset_123 = pd.concat([pFP_APFPs, dFEAT, dPLEC], axis=1)

	# merge dG_exp column to dataset_123, while excluding fictive perturbations and duplicates:
	
	dG_offsets = pd.read_csv(offsets_path, index_col="Perturbation")
	
	
	dataset_123 = pd.merge(dataset_123, dG_offsets, left_index=True,right_index=True)
	
	dataset_123 = (dataset_123[~dataset_123.index.duplicated()])
		
	print("Built dataset; excluded duplicates. \nThe dimensions of the dataset (123) are " + str(len(dataset_123)) + " rows (i.e. perturbations) and " + str(len(dataset_123.columns)) + " columns (i.e. delta-descriptors).")
	print("Writing to \'./ddG_offset_compiled/dataset_123.csv\'..")

	# write to file:
	if not os.path.exists("./ddG_offset_compiled"):
		os.makedirs("./ddG_offset_compiled")

	dataset_123.to_csv("ddG_offset_compiled/dataset_123.csv", index=True)
	print("#####################################")
	
	


	return dataset_123



def split_dataset(dataset_123_offset):

	# Isolate datasets from collective dataframe:
	ddG_offset_column = dataset_123_offset.loc[:, "ddG_offset"]
	pFP_columns = dataset_123_offset.loc[:, :"pfp255"]
	dFEAT_columns = dataset_123_offset.loc[:, "nAcid":"Zagreb1"]
	dPLEC_columns = dataset_123_offset.loc[:, "plec0": ].drop("ddG_offset", axis=1)

	# Construct individual dataframes and name them, perts are indices, 
	# final column is ddG_offset, 1 = pFP, 2 = dFEAT, 3 = dPLEC:
	dataset_1 = pd.merge(pd.DataFrame(pFP_columns), pd.DataFrame(ddG_offset_column), 
		left_index=True,right_index=True)
	dataset_1.name = "dataset_1"
	dataset_2 = pd.merge(pd.DataFrame(dFEAT_columns), pd.DataFrame(ddG_offset_column), 
		left_index=True,right_index=True)
	dataset_2.name = "dataset_2"
	dataset_3 = pd.merge(pd.DataFrame(dPLEC_columns), pd.DataFrame(ddG_offset_column), 
		left_index=True,right_index=True)
	dataset_3.name = "dataset_3"

	# Construct paired dataframes and name them:
	dataset_12_nodG_exp = pd.merge(pd.DataFrame(pFP_columns), pd.DataFrame(dFEAT_columns), 
		left_index=True,right_index=True)
	dataset_12 = pd.merge(dataset_12_nodG_exp, pd.DataFrame(ddG_offset_column), 
		left_index=True,right_index=True)
	dataset_12.name = "dataset_12"

	dataset_13_nodG_exp = pd.merge(pd.DataFrame(pFP_columns), pd.DataFrame(dPLEC_columns), 
		left_index=True,right_index=True)
	dataset_13 = pd.merge(dataset_13_nodG_exp, pd.DataFrame(ddG_offset_column), 
		left_index=True,right_index=True)
	dataset_13.name = "dataset_13"

	dataset_23_nodG_exp = pd.merge(pd.DataFrame(dFEAT_columns), pd.DataFrame(dPLEC_columns), 
		left_index=True,right_index=True)
	dataset_23 = pd.merge(dataset_23_nodG_exp, pd.DataFrame(ddG_offset_column), 
		left_index=True,right_index=True)
	dataset_23.name = "dataset_23"

	# Write and return individual files:
	if not os.path.exists("./ddG_offset_compiled"):
		os.makedirs("./ddG_offset_compiled")

	for file in [dataset_1, dataset_2, dataset_3, dataset_12, dataset_13, dataset_23]:
		print("Writing to \'./ddG_offset_compiled/"+file.name+".csv\'..")
		file.to_csv("ddG_offset_compiled/"+file.name+".csv", index=True)
	print("#####################################")


dataset_123 = build_dataset(pFP_path, dFEAT_path, dPLEC_path, offsets_path)



split_dataset(dataset_123)













