import numpy as np,json,sys
B=np.load(sys.argv[1]); A=json.load(open('mw_ans.json'))
for kind in ('per','obj'):
    for half in (0,1):
        sel=[a for i,a in enumerate(A) if a[2]==kind and (i>=len(A)//2)==half]
        print(sys.argv[1],kind,'2nd' if half else '1st','answer bits %.2f'%np.mean([B[p:p+l].sum() for p,l,_ in sel]))
