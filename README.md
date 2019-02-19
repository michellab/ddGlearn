# ddGlearn

Repository of scripts to generate (eventually) a machine-learned correction term for alchemical free energy calculations.
Scripts pull ligand data from datasets/input/[target]/poses/* and datasets/input/[target]protein/[target]/*. Note that each builder script requires strict adherence to directory formatting in datasets/input/.

###### Prerequisites:
- System: 
	- Python 3.x, OBabel
- Python: 
	- RDKit, DeepChem, Mordred, ODDT, MDTraj
	- Tensorflow, Scikit-Optimize

###### To do:
- pFP:
	- adjust to take input parameters
- dDESCR:
	- adjust to take input parameters
- dPLEC:
	- adjust to take input parameters
	- create PLEC-time matrix

- General:
	- work towards first learned model
	- integrate scripts into a notebook 

