import pandas as pd
import csv
import os

pFP_path = '../pFP/dFP_output/perts_APFPs.csv'
dFEAT_path = '../dFEAT/dFeatures_output/deltaFeatures.csv'
dPLEC_path = '../dPLEC/dPLECs_output/perts_dPLECs.csv'
experi_path = '../ddGs/experimental/experimental_dGs.csv'

def build_dataset(pFP_path, dFEAT_path, dPLEC_path, experi_path):
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

## experimental data extraction ##
	# open and process experimental dGs as dict:
	with open(experi_path, "r") as csvfile:
		experi = []
		reader = csv.reader(csvfile)
		for row in reader:
			experi.append(row)
	experi_dict = { k[0]:float(k[1]) for k in experi }

	# prepare nested list to map experimental values to:
	perts = pFP.index.values.tolist()
	ligpairs = [ pert.split(">") for pert in perts ]

	# map experimental values to perturbations
	ddG_per_pert = []
	for pair in ligpairs:
		ligAdG = experi_dict.get(pair[0])
		ligBdG = experi_dict.get(pair[1])
	# compute experimental ddG (if a member is not in experimental 
	# this is assumed to be a fictive intermediate):
		try:
			ddG = float(ligBdG) - float(ligAdG)
		except TypeError:
			ddG = "fictive"
		pert = pair[0] + ">" + pair[1]
		try:
			ddG_per_pert.append([pert, round(float(ddG), 4)])
		except ValueError:
			ddG_per_pert.append([pert, "fictive"])

	# merge ddG column to dataset_123, while excluding fictive perturbations and duplicates:
	ddG_column = [ pert for pert in ddG_per_pert if pert[1] != "fictive"]
	ddG_df = pd.DataFrame(ddG_column, columns=["Perturbation", "ddG"]).set_index("Perturbation")
	
	dataset_123 = pd.merge(dataset_123, ddG_df, left_index=True,right_index=True)
	dataset_123 = (dataset_123[~dataset_123.index.duplicated()])
	print("Built dataset; excluded fictive perturbations and duplicates. \nThe dimensions of the data set are " + str(len(dataset_123)) + " rows (i.e. perturbations) and " + str(len(dataset_123.columns)) + " columns (i.e. "u"\N{GREEK CAPITAL LETTER DELTA}""descriptors).")
	print("Writing to \'./datasets/dataset_123.csv\'..")

	# write to file:
	if not os.path.exists("./datasets"):
		os.makedirs("./datasets")

	dataset_123.to_csv("datasets/dataset_123.csv", index=True)

build_dataset(pFP_path, dFEAT_path, dPLEC_path, experi_path)












