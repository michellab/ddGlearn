import os
import csv
import itertools


with open("../../morph.in", 'rt') as morph_file:

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
        
    second_cleaned = (first_cleaned.replace(" ", "").replace(">",", "))
    molecule_pairs = second_cleaned.split("\n")
    perturbation_list = []
    for pert in molecule_pairs:
        if len(pert) != 0: 
            pert_pair = pert.split(", ", 1)
            if "_BM2" not in pert_pair[0] and "_BM2" not in pert_pair[1]:
            	perturbation_list.append(pert_pair)
    print("Amount of perturbations:",len(perturbation_list))
    print("#####################################")
    cleaned_perts = []
    for pert in perturbation_list:
    	member1 = pert[0].replace("_BM1", "").replace("_","")
    	member2 = pert[1].replace("_BM1", "").replace("_","")
    	cleaned_perts.append( str(member1)+">"+str(member2))

print(cleaned_perts)

with open("IC50s.csv", 'rt') as ic50_file:
	experi = []
	reader = csv.reader(ic50_file)
	for row in reader:
		experi.append(row[0].split("\t"))

#print(experi)