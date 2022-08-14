%reset -f
#%%
import calliope
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import matplotlib.pyplot as plt
import re


calliope.set_log_verbosity('INFO')
opt = SolverFactory('gurobi')

# Load models
model_simple = calliope.Model('../../1/simple/model.yaml')

#creating a variable to store the dual variables
model_simple.run(build_only=True)
model_simple._backend_model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
results = opt.solve(model_simple._backend_model) # also load results to model

# display all dual variables
#duals from the simple model
duals = {} #print ("Duals")
for a in model_simple._backend_model.component_objects(pyo.Constraint, active=True):
    duals[("{} Constraint".format(a))] = []
    for index in a:
        duals["{} Constraint".format(a)].append(("{}".format(index), model_simple._backend_model.dual[a[index]]))
    duals["{} Constraint".format(a)] = pd.DataFrame(duals["{} Constraint".format(a)])


# REORGANIZING THE DATAFRAME
system_balance_duals = duals['system_balance_constraint Constraint']

def split_into_info3(nonsense_frame3):
    column3=nonsense_frame3[0]
    new_frame3 = pd.DataFrame(index=range(len(column3)), columns=['region','carrier','timestep','dual-value'])
    new_frame3['dual-value'] = new_frame3['dual-value'].astype(float)
    k=0
    for row3 in column3:
        info3 = set(re.split("[(::)]",row3))
        info3.remove('')
        info3 = sorted(list(info3))                
        new_frame3['region'].loc[k] = info3[0].split("'")[-1]
        new_frame3['carrier'].loc[k], timestep = (info3[2].split("'")[0],info3[2].split("'")[-1])
        new_frame3['timestep'].loc[k] = pd.date_range(timestep +':00:00',timestep +':00:00',freq='H')[0]
        new_frame3['dual-value'].loc[k] = float(nonsense_frame3.loc[k][1])
        
        k += 1

    return (new_frame3)

balance_duals = split_into_info3(system_balance_duals)


# running the models
model_simple.run()


for region in ['region1']:
    balance_region = balance_duals[(balance_duals['region'] == region)]
    plt.plot(balance_region['timestep'], balance_region['dual-value'], label=(region+' system balance duals'))
plt.plot(0.01* model_simple.get_formatted_array('carrier_con').loc[{'techs':'demand_power', 'carriers':'power'}].sum('locs').to_pandas(),label=('demand power'))
plt.ylim(ymin=0,ymax=0.1)
plt.legend(loc='center left')
plt.axhline(y=0, color='k',linewidth=0.5)


# model_simple.plot.timeseries()


#%%thesis report plots

# create figure and axis objects with subplots()

fig,ax = plt.subplots()
plt.title('Simple model: Power duals vs power demand')
# make a plot
for region in ['region1']:
    balance_region = balance_duals[(balance_duals['region'] == region)]
    data1=plt.plot(balance_region['timestep'], balance_region['dual-value'],color="black",label=('shadow price power'))
    plt.xticks(rotation=90)
    plt.yscale("log")
    # plt.ylim(ymin=0)
    # plt.legend(loc='upper left')
# set y-axis label
ax.set_xlabel("Timestep")
ax.set_ylabel("Shadow price in €/kWh")
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
data2=plt.plot(-0.001*model_simple.get_formatted_array('carrier_con').loc[{'techs':'demand_power', 'carriers':'power'}].sum('locs').to_pandas(),label=('power demand'),color="blue")
data3=plt.plot(0.001*model_simple.get_formatted_array('carrier_prod').loc[{'techs':'ccgt', 'carriers':'power'}].sum('locs').to_pandas(),label=('power production ccgt'),color="red")

ax2.set_ylabel("Power in kW")

data = data1+data2+data3
labs = [l.get_label() for l in data]
ax.legend(data, labs, loc='upper left')
# plt.legend()
plt.show()


# save the plot as a file

# plt.plot(-model_simple.get_formatted_array('carrier_prod').loc[{'techs':'ccgt', 'carriers':'power'}].sum('locs').to_pandas(),label=('ccgt power production'))

