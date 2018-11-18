import pandas as pd
import glob
import functools

stockinfo = {}

for infile in glob.glob("./stockinfo/*.csv"):
    stockinfo[infile] = pd.read_csv(infile)

allstockinfo = functools.reduce((lambda x,y: pd.concat([x,y])), stockinfo.values())

print ("DEBUG: print all stock dataframe:")
print (allstockinfo)
