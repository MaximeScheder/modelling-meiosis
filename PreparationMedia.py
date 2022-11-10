# -*- coding: utf-8 -*-
"""
This script will do inference on a simple test-landscape, in order to see
if the computer is able to handle it.
"""
import pandas as pd
# In order : Ac (i,f) - G (i,f) - N (i,f)
X_init = [10, 10, 5]
c = pd.read_excel("C:/Users/msche/OneDrive/Documents/EPFL/LPBS/Yeast/Data/Nutrients_experiment/experiment-design.xlsx")
V_tot = 3
N = c.index[-1]+1
v_stupid = 0

print("Volume of YPD cell needed : {:.2f} ml".format(N*0.24))
print("Volume of BYTA needed : {:.2f} ml".format(N*14))
for n in c.index:
    
    
    V = V_tot/5 # for YNB
    #v_stupid = 3/5
    print("{}. YNB - {}xAc {}xG {}xN :".format(n+1, c.Acetate[n], c.Glucose[n], c.Nitrogen[n]))
    print("\t {:.1f} ml YNB".format(V))
    
    v = V_tot/X_init[0]*c.Acetate[n]
    #v_stupid += 3/X_init[0]*c.Acetate[n]
    V += v
    
    
    print("\t {:.1f} ul Ac".format(v*1000))

    v = V_tot/X_init[1]*c.Glucose[n]
    v_stupid += 3/X_init[1]*c.Glucose[n]

    V += v
    print("\t {:.1f} ul G".format(v*1000))

    v = V_tot/X_init[2]*c.Nitrogen[n]
    v_stupid += 3/X_init[2]*c.Nitrogen[n]
    V += v
    print("\t {:.1f} ul N".format(v*1000))
        
    


    print("\t {:.2f} ml H2O".format(V_tot-V))
    
    
    if n !=N-1:
        input("Press any key for next flask \n")
    else:
        print("Done")
    


    
