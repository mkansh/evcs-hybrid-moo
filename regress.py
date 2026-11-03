import sys, numpy as np, os
sys.path.insert(0, sys.argv[1]); os.environ['EVCS_ENC']=sys.argv[2]
import model, algos
from model import *
from algos import *
from algos_extra import nsga3, mopso
sc=make_scenarios(); out=[]
for f in (lambda p: nsga2(p,300,False), lambda p: nsga2(p,300,True), lambda p: moead(p,300), lambda p: nsga3(p,300), lambda p: mopso(p,300)):
    p=Problem(sc); X,F,cv,tr=f(p); out.append(F)
np.save(sys.argv[3], np.concatenate([o.ravel() for o in out]))
