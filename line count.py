import matplotlib.pyplot as plt
import matplotlib.colors as cm
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import numpy as npimport matplotlib.pyplot as plt
import matplotlib.colors as cm
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import numpy as np

BETA = 2e-2
LAMBDA = 0.1
vt = 0.7

def f1(x,y,l):  #IDS
    return  BETA*((x-vt)*y-1/2*pow(y,2)) * (1+l*y)

def f2(x,y,l):  #IDS
    return 0.5*BETA*pow(x-vt,2) * (1+l*y)

# def f4(x,y,l):
#     vt=0.7
#     ret = f1(x,y,l)
#     for i in range(len(y)):
#             if x[i]-vt>y[i]:
#                 ret[i] = f2(x[i],y[i],l)
#     return ret

def f5(x,y,l):
#     factor=1
#     if y<0:
#         y = -y
#         factor=-1
    if (x-vt > y):
#         if (x == 2 and y == -5.0):
#             print(f1(x,y,l) + (y-9)/3e3)
        return (f1(x,y,l) + (y-9)/3e3)#*factor
    else:
#         if (x == 2 and y == -5.0):
#             print(f1(x,y,l) + (y-9)/3e3)
        return (f2(x,y,l) + (y-9)/3e3)#*factor

def f5d1(x,y,l):
#     factor=1
#     if y<0:
#         y = -y
#         factor=-1
    if (x-vt > y):
        return BETA*y*(1+l*y)#*factor
    else:
        return BETA*(x-vt)*(1+l*y)#*factor

def f5d2(x,y,l):
#     factor=1
#     if y<0:
#         y = -y
#         factor=-1
    if (x-vt>y):
        return (BETA*((x-vt)+2*l*y*(x-vt)-y-3/2*l*pow(y,2)) + 1/3e3)#*factor
    else:
        return (BETA/2*l*pow(x-vt,2) + 3/1e3)#*factor

def f6(x,y,l):
    return (x-9)/1e3+x/2e3

def f6d1(x,y,l):
    return 1/1e3 + 1/2e3

def f6d2(x,y,l):
    return 0

def findDelta(x,y,l=LAMBDA):
    f1x = f5d1(x,y,l)
    f1y = f5d2(x,y,l)
    f2x = f6d1(x,y,l)
    f2y = f6d2(x,y,l)
    invmatrix = np.linalg.inv(np.array([[f1x,f1y],
                                        [f2x,f2y]]))
    fmatrix = np.array([f5(x,y,l),f6(x,y,l)])
    return -np.matmul(invmatrix,fmatrix)

def findFunction(x,y,l=LAMBDA):
    return np.array([f5(x,y,l),f6(x,y,l)])
absdij

X, Y = np.meshgrid(np.arange(-1, 15, 0.1,dtype='float64'), np.arange(-11, 11, 0.1,dtype='float64'))
Xp, Yp = np.meshgrid(np.arange(-1, 16, 2,dtype='float64'), np.arange(-11, 12, 2,dtype='float64'))

x_shape = X.shape
xp_shape = Xp.shape

F1 = np.zeros_like(X)
F1X = np.zeros_like(X)
F1Y = np.zeros_like(X)

F2 = np.zeros_like(X)
F2X = np.zeros_like(X)
F2Y = np.zeros_like(X)

F3X = np.zeros_like(X)
F3Y = np.zeros_like(X)

for i in range(len(X)):
    for j in range(len(X[i])):
        x = X[i][j]
        y = Y[i][j]
        F1[i][j] = f5(x,y,LAMBDA)
        if (x == 2 and y == -5):
            print(F1[i][j])
        F2[i,j] = f6(x,y,LAMBDA)


for i in range(x_shape[0]):
    for j in range(x_shape[1]):
        F1X[i,j] = f5d1(X[i,j],Y[i,j],LAMBDA)
        F1Y[i,j] = f5d2(X[i,j],Y[i,j],LAMBDA)
        F2X[i,j] = f6d1(X[i,j],Y[i,j],LAMBDA)
        F2Y[i,j] = f6d2(X[i,j],Y[i,j],LAMBDA)

        invmatrix = np.linalg.inv(np.array([[F1X[i,j],F1Y[i,j]],
                              [F2X[i,j],F2Y[i,j]]]))
        F3X[i,j],F3Y[i,j] = -np.matmul(invmatrix,np.array([F1[i,j],F2[i,j]]))








F1p = np.zeros_like(Xp)
F1Xp = np.zeros_like(Xp)
F1Yp = np.zeros_like(Xp)

F2p = np.zeros_like(Xp)
F2Xp = np.zeros_like(Xp)
F2Yp = np.zeros_like(Xp)

F3Xp = np.zeros_like(Xp)
F3Yp = np.zeros_like(Xp)

for i in range(len(Xp)):
    for j in range(len(Xp[i])):
        x = Xp[i][j]
        y = Yp[i][j]
        F1p[i][j] = f5(x,y,LAMBDA) + (y-9)/1e3
        F2p[i,j] = f6(x,y,LAMBDA)


for i in range(xp_shape[0]):
    for j in range(xp_shape[1]):
        F1Xp[i,j] = f5d1(Xp[i,j],Yp[i,j],LAMBDA)
        F1Yp[i,j] = f5d2(Xp[i,j],Yp[i,j],LAMBDA)
        F2Xp[i,j] = f6d1(Xp[i,j],Yp[i,j],LAMBDA)
        F2Yp[i,j] = f6d2(Xp[i,j],Yp[i,j],LAMBDA)

        invmatrix = np.linalg.inv(np.array([[F1Xp[i,j],F1Yp[i,j]],
                              [F2Xp[i,j],F2Yp[i,j]]]))
        F3Xp[i,j],F3Yp[i,j] = -np.matmul(invmatrix,np.array([F1p[i,j],F2p[i,j]]))

mag = np.sqrt(F3Xp*F3Xp + F3Yp*F3Yp)
quiverX = F3Xp/mag
quiverY = F3Yp/mag
color = np.log(mag,np.zeros_like(mag)+2)
color

fig,ax = plt.subplots()


#colormap
cmap = cm.ListedColormap(plt.cm.plasma(10), "name")



plt.rcParams['contour.negative_linestyle'] = 'solid'
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"

def convexity1(x,y):
    if y < x-0.7:
        return 0
    else:
        return BETA+BETA*LAMBDA*y

def convexity2(x,y):
    if y < x-0.7:
        return 2*LAMBDA*BETA*x-1.4*LAMBDA*BETA-3*LAMBDA*BETA*y-BETA
    else:
        return 0

def derivative1(x,y):
    if (x-vt > y):
        return BETA*y*(1+LAMBDA*y)#*factor
    else:
        return BETA*(x-vt)*(1+LAMBDA*y)#*factor

def derivative2(x,y):
    l=LAMBDA
    if (x-vt>y):
        return 1/3000+(BETA*((x-vt)+2*l*y*(x-vt)-y-3/2*l*pow(y,2)) + 1/3e3)#*factor
    else:
        return 1/3000+(BETA/2*l*pow(x-vt,2) + 3/1e3)#*factor

Sign1 = np.zeros_like(X)
Sign2 = np.zeros_like(X)
for i in range(len(X)):
    for j in range(len(X[0])):
        Sign1[i,j] = convexity1(X[i,j],Y[i,j]) * derivative1(X[i,j],Y[i,j])/np.abs(derivative1(X[i,j],Y[i,j]))
        Sign2[i,j] = convexity2(X[i,j],Y[i,j])* derivative2(x,y)/np.abs(derivative2())

print(Sign1[0,0])
print(Sign2[0,0])

normCS = cm.Normalize(vmin=-0.1,vmax=0.1,clip=False)
CS = ax.contourf(X,Y,Sign2)#levels=np.arange(-2,10,0.5))  #x-axis: e1, y-axis: e2
# ticks=np.arange(-0.1,0.1,0.05)
cbCS = plt.colorbar(CS, format='%.1f', norm=normCS)
cbCS.ax.get_yaxis().labelpad = 15
cbCS.ax.set_ylabel('# of contacts', rotation=270)




# ax.clabel(CS, fontsize=8, inline=True)

plt.xlim(-0.5,15)
plt.ylim(-11,11)

plt.title('Title',fontsize=10)


ax.set_aspect='equal'
plt.savefig('circuit1.png',dpi=1000)

plt.set_cmap('bwr')
vmin = -0.1
vmax = 0.1
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
plt.pcolor(X, Y, Sign1,vmin=vmin, vmax=vmax, norm=norm)
plt.colorbar()
plt.xlabel('$e_1 (V)$')
plt.ylabel('$e_2 (V)$')
plt.savefig('curvature1.png',ppi=1000)

plt.set_cmap('bwr')
vmin = -0.1
vmax = 0.1
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
plt.pcolor(X, Y, Sign2,vmin=vmin, vmax=vmax, norm=norm)
plt.colorbar()
plt.xlabel('$e_1 (V)$')
plt.ylabel('$e_2 (V)$')
plt.savefig('curvature2.png',ppi=1000)

fig,ax = plt.subplots()


E2 = np.arange(-10,10,0.1)
E1 = np.zeros_like(E2)+5
Ids = []
for i in range(len(E1)):
    Ids.append(f5(E1[i],E2[i],LAMBDA))

Ids = np.array(Ids)

plt.plot(E2,Ids)

E2 = np.arange(0,10,0.1)
E1 = np.zeros_like(E2)+5
Ids = []
for i in range(len(E1)):
    Ids.append(f5(E1[i],E2[i],0.1))

Ids = np.array(Ids)

#plt.plot(E2,Ids)

F3Xp[4,13], F3Yp[4,13]

f5d1(3,-6,0.1)

f5d2(3,-6,0.1)

#Xp,Yp

F2[5,12]

BETA*((2-vt)*(-5)-1/2*pow(-5,2)) * (1+0.1*(-5)) + (-5-9)/3e3

X[5,12],Y[5,12]

f1(2,-5,0.1)

F1X[5,12],F1Y[5,12]

EPSILON = 0.3

posp = np.array([3.,-3.])
pos = np.copy(posp) + 2
functionp = findFunction(pos[0],pos[1])
function = functionp + 2
print(posp)
while np.abs(pos[0]-posp[0])>1e-6 or np.abs(pos[0]-posp[0])>1e-6:
    pos = posp
    posp = posp + EPSILON * findDelta(posp[0],posp[1])
    function = functionp
    functionp = findFunction(pos[0],pos[1])
    print(posp,functionp)

#iterations vs scaling factor without momentum

epsilons = np.arange(0.025,0.5,0.025)
iterations = []
starting_pos = np.array([0.,0.])
for epsilon in epsilons:
    posp = np.array([0.,0.])
    pos = np.copy(posp) + 2
    cnt = 0
    while (np.abs(pos[0]-posp[0])>1e-6 or np.abs(pos[0]-posp[0])>1e-6) and cnt < 10000:
        pos = posp
        posp = posp + epsilon * findDelta(posp[0],posp[1])
        cnt += 1
    if pos[1] > 0:
        iterations.append(cnt)
    else:
        iterations.append(None)
iterations = np.array(iterations)
plt.plot(epsilons[0:len(iterations)],iterations)
plt.xlim(0,0.5)
plt.xlabel('$a$')
plt.ylabel('# of iterations')
iterations
plt.savefig('circuit2.png',dpi=1000)



#iterations vs scaling factor with momentum

epsilons = np.arange(0.025,0.5,0.025)
# momentum_factors = np.arange(0.025,0.9,0.025)
momentum_factors = np.arange(0,0.6,0.1)
iterationsM = []
starting_pos = np.array([0.,0.])

row = -1
for epsilon in epsilons:
    row += 1
    iterationsM.append([])
    for momentum_factor in momentum_factors:
        posp = np.array([0.,0.])
        pos = np.copy(posp) + 2
        v = 0
        cnt = 0
        while (np.abs(pos[0]-posp[0])>1e-6 or np.abs(pos[0]-posp[0])>1e-6) and cnt < 10000:
            v = momentum_factor * v + findDelta(posp[0],posp[1])
            pos = posp
            posp = posp + epsilon * v
            cnt += 1
        if pos[1] > 0:
            iterationsM[row].append(cnt)
        else:
            iterationsM[row].append(None)
    iterationsM[row] = np.array(iterationsM[row],dtype='object')

iterationsM = np.array(iterationsM)
# plt.plot(epsilons[0:len(iterationsM)],iterationsM,c='k')
# iterationsM.shape
# [iterationsM[i].shape for i in range(len(iterationsM))]

#momentum test

EPSILON = 0.2
MOMENTUM_FACTOR1 = 0.
MOMENTUM_FACTOR2 = 0.3

posp1 = np.array([3.,3.])
pos1 = np.copy(posp1) + 2
pos1store = []
pos1store.append(posp1)
v = 0
print(posp1)
cnt1 = 0
functionp1 = findFunction(pos1[0],pos1[1])
function1 = functionp1 + 2
while np.abs(pos1[0]-posp1[0])>1e-6 or np.abs(pos1[1]-posp1[1])>1e-6:
    v = MOMENTUM_FACTOR1 * v + findDelta(posp1[0],posp1[1])
    pos1 = posp1
    posp1 = posp1 + EPSILON * v
    function1 = functionp1
    functionp1 = findFunction(pos1[0],pos1[1])
    cnt1 += 1
    print(posp1,functionp1)
    pos1store.append(posp1)
print(cnt1)

posp2 = np.array([3.,3.])
pos2 = np.copy(posp2) + 2
pos2store = []
pos2store.append(posp2)
v = 0
print(posp2)
cnt2 = 0
functionp2 = findFunction(pos2[0],pos2[1])
function2 = functionp2 + 2
while np.abs(pos2[0]-posp2[0])>1e-6 or np.abs(pos2[1]-posp2[1])>1e-6:
    v = MOMENTUM_FACTOR2 * v + findDelta(posp2[0],posp2[1])
    pos2 = posp2
    posp2 = posp2 + EPSILON * v
    function2 = functionp2
    functionp2 = findFunction(pos2[0],pos2[1])
    cnt2 += 1
    print(posp2,functionp2)
    pos2store.append(posp2)
print(cnt2)

fig,ax = plt.subplots()


#colormap
norm = cm.Normalize(vmin=0,vmax=50,clip=False)
cmap = cm.ListedColormap(plt.cm.plasma(np.linspace(0.25,1,10)), "name")


plt.rcParams['contour.negative_linestyle'] = 'solid'
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"


normCS = cm.Normalize(vmin=-0.4,vmax=1.2,clip=False)
CS = ax.contour(X,Y,F1)#levels=np.arange(-2,10,0.5))  #x-axis: e1, y-axis: e2
#divnorm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
#cmap2 = cm.ListedColormap(plt.cm.RdBu(np.linspace(0.25,1,10)), "name")
normCS2 = cm.Normalize(vmin=-0.005,vmax=0.005,clip=False)
CS2 = ax.contour(X,Y,F2,colors='k',alpha=0.3)#cmap=plt.cm.PuOr,norm=norm2,)#,levels=np.arange(-2,10,0.5))  #x-axis: e1, y-axis: e2

#q = ax.quiver(Xp, Yp, F3Xp, F3Yp, units='xy',color='red',scale=1)
q = ax.quiver(Xp, Yp, quiverX, quiverY,mag,units='xy',scale=1,cmap=cmap,norm=norm)
cb = plt.colorbar(q, ticks=np.arange(0,51,10), format='%.1f', norm=norm)
cb.ax.get_yaxis().labelpad = 15
cb.ax.set_ylabel('# of contacts', rotation=270)
# q = ax.quiver(X, Y, F2X, F2Y, units='xy' ,scale=1, color='blue')

XYposp1 = np.array(pos1store[0:len(pos1store)-1]).transpose()
Qposp1 = np.array(pos1store[1:len(pos1store)]).transpose()
Qposp1 = Qposp1-XYposp1
qR = ax.quiver(XYposp1[0],XYposp1[1],Qposp1[0],Qposp1[1],units='xy',scale=1)

cbCS = plt.colorbar(CS, ticks=np.arange(-0.8,2.5,0.4), format='%.1f', norm=normCS)
cbCS.ax.get_yaxis().labelpad = 15
cbCS.ax.set_ylabel('# of contacts', rotation=270)


#ax.clabel(CS, fontsize=8, inline=True)
ax.clabel(CS2, fontsize=8, inline=True)
# ax.contour(X,Y,F2,levels=np.arange(-20,20,4),colors = 'blue')

# ax.set_aspect('equal')

plt.xlim(-0.5,15)
plt.ylim(-11,11)

plt.title('Title',fontsize=10)


ax.set_aspect='equal'
plt.savefig('circuit1.png',dpi=1000)

XYposp1, Qposp1

fig,ax = plt.subplots()


#colormap
norm = cm.Normalize(vmin=0,vmax=50,clip=False)
cmap = cm.ListedColormap(plt.cm.plasma(np.linspace(0.25,1,10)), "name")


plt.rcParams['contour.negative_linestyle'] = 'solid'
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"

plt.set_cmap('viridis')

normCS = cm.Normalize(vmin=-0.4,vmax=1.2,clip=False)
CS = ax.contour(X,Y,F1)#levels=np.arange(-2,10,0.5))  #x-axis: e1, y-axis: e2
#divnorm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
#cmap2 = cm.ListedColormap(plt.cm.RdBu(np.linspace(0.25,1,10)), "name")
normCS2 = cm.Normalize(vmin=-0.005,vmax=0.005,clip=False)
CS2 = ax.contour(X,Y,F2,colors='k',alpha=0.3)#cmap=plt.cm.PuOr,norm=norm2,)#,levels=np.arange(-2,10,0.5))  #x-axis: e1, y-axis: e2

#q = ax.quiver(Xp, Yp, F3Xp, F3Yp, units='xy',color='red',scale=1)
q = ax.quiver(Xp, Yp, quiverX, quiverY,mag,units='xy',scale=1,cmap=cmap,norm=norm)
cb = plt.colorbar(q, ticks=np.arange(0,51,10), format='%.1f', norm=norm)
cb.ax.get_yaxis().labelpad = 15
cb.ax.set_ylabel('$\Delta e$ vector field', rotation=270)
# q = ax.quiver(X, Y, F2X, F2Y, units='xy' ,scale=1, color='blue')

XYposp2 = np.array(pos2store[0:len(pos2store)-1]).transpose()
Qposp2 = np.array(pos2store[1:len(pos2store)]).transpose()
Qposp2 = Qposp2 - XYposp2
# qR = ax.quiver(XYposp2[0],XYposp2[1],Qposp2[0],Qposp2[1],units='xy',scale=1)

cbCS = plt.colorbar(CS, ticks=np.arange(-0.8,2.5,0.4), format='%.1f', norm=normCS)
cbCS.ax.get_yaxis().labelpad = 15
cbCS.ax.set_ylabel('$f_1(e)$', rotation=270)


#ax.clabel(CS, fontsize=8, inline=True)
ax.clabel(CS2, fontsize=8, inline=True)
# ax.contour(X,Y,F2,levels=np.arange(-20,20,4),colors = 'blue')

# ax.set_aspect('equal')

plt.xlim(-0.5,15)
# plt.xlim(5,7.5)
plt.ylim(-11,11)
# plt.ylim(-1,1)


ax.set_aspect='equal'
plt.xlabel('$e_1 (V)$')
plt.ylabel('$e_2 (V)$')
plt.savefig('circuit1.png',dpi=1000)

XYposp2, Qposp2

plt.plot([i for i in range(len(XYposp2[0]))],XYposp2[0])#,label='$e_1$ convergence with momentum')
plt.plot([i for i in range(len(XYposp2[1]))],XYposp2[1],label='$e_2$ convergence with momentum')
plt.plot([i for i in range(len(XYposp1[0]))],XYposp1[0])#,label='$e_1$ convergence without momentum')
plt.plot([i for i in range(len(XYposp1[1]))],XYposp1[1],label='$e_2$ convergence without momentum')
plt.xlim(0,40)
plt.legend()
plt.ylim(-0.1,0.1)
plt.xlabel('# of iterations')
plt.ylabel('$e (V)$')
plt.savefig('sample_case2',ppi=1000)
# plt.ylim(0.01,0.04)




plt.plot([i for i in range(len(XYposp1[0]))],XYposp1[0])
plt.plot([i for i in range(len(XYposp1[1]))],XYposp1[1])
# plt.xlim(10,)
plt.ylim(-1,1)

#graph iterations vs momentum factor with momentum

plt.plot(momentum_factors[0:len(iterationsM[0])],iterationsM[0],c='k')  #EPSILON=0.025
plt.plot(momentum_factors[0:len(iterationsM[1])],iterationsM[1],c='b')
plt.plot(momentum_factors[0:len(iterationsM[5])],iterationsM[5],c='r')
plt.plot(momentum_factors[0:len(iterationsM[10])],iterationsM[10],c='g')
plt.plot(momentum_factors[0:len(iterationsM[13])],iterationsM[13],c='c')
plt.ylim(-5,150)



#graph iterations vs scaling factor with momentum

iterationsE = iterationsM.transpose()

plt.plot(epsilons[0:len(iterationsE[0])],iterationsE[0],c='orange',label='momentum=0')  #MOMENTUM=0.
plt.plot(epsilons[0:len(iterationsE[1])],iterationsE[1],c='b',label='momentum=0.1')  #MOMENTUM=0.1
plt.plot(epsilons[0:len(iterationsE[2])],iterationsE[2],c='r',label='momentum=0.2')  #MOMENTUM=0.2
plt.plot(epsilons[0:len(iterationsE[3])],iterationsE[3],c='g',label='momentum=0.3')  #MOMENTUM=0.3
plt.ylim(-5,100)
plt.xlim(0,0.5)
plt.xlabel('a')
plt.legend()
plt.ylabel('# of iterations')
plt.savefig('circuit3.png',dpi=1000)

iterationsM.shape

X = np.arange(0,6,1)
Y = np.array([1,3,4,None,2,4])
plt.plot(X,Y)
X,Y

