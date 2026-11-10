#!/usr/bin/env python3
"""Per input range ulp error of f32mathcli answers: tools/f32math_split.py XS ANSWERS ORACLE."""
import struct,sys
b2f=lambda s:struct.unpack("<f",struct.pack("<I",int(s,16)))[0]
def o(s):
    b=int(s,16); return -(b&0x7FFFFFFF) if b>>31 else b
xs=[l.strip() for l in open(sys.argv[1])];A=[l.split() for l in open(sys.argv[2])];O=[l.split() for l in open(sys.argv[3])]
cls={"exp softmax [-30,0]":(0,lambda x:-30<=x<=0),"exp full (-104,89)":(0,lambda x:-104<x<89),"log (0,1] probs":(1,lambda x:0<x<=1),"log near 1":(1,lambda x:abs(x-1)<1e-3),"log normal pos":(1,lambda x:1.2e-38<x<3.4e38)}
for k,(j,f) in cls.items():
    ds=[]
    for s,a,r in zip(xs,A,O):
        x=b2f(s)
        if x==x and f(x):
            ds.append(float('inf') if 'nan' in (a[j],r[j]) else abs(o(a[j])-o(r[j])))
    n=len(ds);print(f"{k:24} n={n:6} exact {100*sum(d==0 for d in ds)/n:6.2f}%  >1ulp {sum(d>1 for d in ds):5}  max {max(ds):g}")
