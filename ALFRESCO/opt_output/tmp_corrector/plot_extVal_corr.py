import csv
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy import stats
from sklearn.metrics import mean_absolute_error
import numpy as np
import seaborn as sns

with open("TopPerformer_externalVal_df.csv", "r") as file:
	reader = csv.reader(file)
	data = [ row for row in reader ]

result = pd.DataFrame(data[1:], columns=data[0])

print(result)

exp = [float(val) for val in result.iloc[:,4].values]

fep = [float(val) for val in result.iloc[:,3].values]

fep_corr = [float(val) for val in result.iloc[:,5].values]

slope, intercept, r_value, p_value, std_err = stats.linregress(exp, fep_corr)
mae = mean_absolute_error(exp, fep_corr)

print("R2-Score: "+str(r_value**2))
print("MAE: "+str(mae))

print(result)
sns.set()
plt.scatter(x=fep_corr, y=exp)

mn = np.min(exp)
mx = np.max(exp)
x1=np.linspace(mn,mx, 500)
y1=slope*x1+intercept

plt.xlabel("Predicted ddG (kcal/mol)")
plt.ylabel("Experimental ddG (kcal/mol)")

plt.title("Corrected FEP vs Experimental ddG")
plt.text(1.4,-1.6,"R2: "+str(round(r_value**2, 3)))
plt.text(1.4,-1.8,"MAE: "+str(round(mae, 3)))
plt.plot([-2, 2], [-2, 2], linewidth=2, color="crimson")


#plt.plot(x1, y1)
plt.show()