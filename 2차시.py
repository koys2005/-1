def erasemiro(li,x,y):
    change=False
    for i in range(1,x-1):
        for j in range(1,y-1):
            if [i,j]!=[1,1] and [i,j]!=[x-2,y-2] and li[i][j]==0 and li[i+1][j]+li[i-1][j]+li[i][j-1]+li[i][j+1]>2:
                li[i][j]=1
                change=True
    if change==False:
        return
    erasemiro(li,x,y)

def printpath(li,x,y):
    for i in range(x):
        for j in range(y):
            if li[i][j]==1:
                print(k[i][j],end=' ')
            else:
                print(2,end=' ')
        print()
a,b=map(int,input().split())
k=[]
l=[]
for i in range(b):
    p=list(map(int,input().split()))
    q=p[:]
    k.append(p)
    l.append(q)
erasemiro(l,b,a)
printpath(l,b,a)
