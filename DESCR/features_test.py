import pandas as pd 
import csv
from rdkit import ML, Chem

from mordred import Calculator, descriptors




"""
df = pd.read_csv("../dFP/dFP_output/perts_APFPs.csv", nrows=1)

member1_smiles = df["fullmember1"].tolist()
member2_smiles = df["fullmember2"].tolist()

member1_mols = [Chem.MolFromSmiles(smiles) for smiles in member1_smiles]
member2_mols = [Chem.MolFromSmiles(smiles) for smiles in member2_smiles]


calc = Calculator(descriptors, ignore_3D=False)

print("Calculating chemical descriptors for first member: ")
df_1 = calc.pandas(member1_mols)
"""
# consult http://mordred-descriptor.github.io/documentation/master/descriptors.html


with open("./discarded_descriptors.txt", "r") as file:
	descrpts = []
	reader = csv.reader(file)
	for item in file:
		print(item)
		#item.append([descrpts])
print(descrpts)



#print("Calculating chemical descriptors for second member: ")
#df_2 = calc.pandas(member2_mols)


#df_1 = df_1.iloc[:, 0:5]
#df_2 = df_2.iloc[:, 0:5]

#print((df_1 - df_2)/df_1)





