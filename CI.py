import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from hyppo.independence import Hsic
import csv
import pandas as pd
import os



def sn(size):
    return np.random.standard_normal(size)
def su(size):
    return np.random.uniform(0,1,size)
def ts(size,df):
    return np.random.standard_t(df,size)
def genModel(code,Nx,Ny):
    if code=='A':
        X = Nx
        Y = np.multiply(np.multiply(X, X), X) + X + Ny
        return X,Y
    if code=='B':
        X = Nx
        Y = X + Ny
        return X,Y
def interModel(code,Nx,Ny,intervertor,intervertion):
    if intervertion=='x':
        if code=='A':
            X = intervertor
            Y = np.multiply(np.multiply(X, X), X) + X + Ny
            return X,Y
        if code=='B':
            X = intervertor
            Y = X + Ny
            return X,Y
    if intervertion=='y':
        if code=='A':
            X = Nx
            Y = intervertor
            return X,Y
        if code=='B':
            X = Nx
            Y = intervertor
            return X,Y

def prob(vector,number):
    n = len(vector)
    k = 0
    for i in range(n):
        if vector[i]<number+0.02:
            if vector[i]>number-0.02:
                k+=1
    return k/n

def prob2(x,y,number):
    n = len(x)
    k = 0
    for i in range(n):
        if y[i]<(number+0.01):
            if y[i]>(number-0.01):
                k+=1
    vector = np.zeros(k)
    k = 0
    for i in range(n):
        if y[i]<(number+0.01):
            if y[i]>(number-0.01):
                vector[k]=x[i]
                k+=1
    return vector

def prob3(x,y,z,w,number1,number2):
    n = len(x)
    k = 0
    tresh = 0.25
    for i in range(n):
        if z[i]<(number1+tresh):
            if z[i]>(number1-tresh):
                if w[i]< (number2 + tresh):
                    if w[i] > (number2 - tresh):
                        k+=1
    X = np.zeros(k)
    Y = np.zeros(k)
    print(k)
    k = 0
    for i in range(n):
        if z[i] < (number1 + tresh):
            if z[i] > (number1 - tresh):
                if w[i] < (number2 + tresh):
                    if w[i] > (number2 - tresh):
                        X[k]=x[i]
                        Y[k]=y[i]
                        k += 1
    return X,Y

def plotCond(x,y,code,number,first):
    plt.figure()
    x_sorted = np.sort(prob2(x, y, number))
    if len(x_sorted)==0:
        return
    bins = np.linspace(x_sorted[0], x_sorted[-1], 25)
    hist, bins = np.histogram(x_sorted, bins)
    p = prob(y, number)
    hist = hist / (len(x_sorted)) / p
    hist = hist / sum(hist)
    widths = np.diff(bins)
    plt.bar(bins[:-1], hist, widths)
    plt.title("Model " + code + " P("+first+"|"+str(number)+")")
    plt.show()


def findDist(X,Y,type,Nx,Ny,model):
    code = str(type)
    x = X
    y = Y
    plt.figure()
    plt.hist(x,density=True)
    plt.title("Model "+code+" P(x)")
    plt.show()
    plt.figure()
    plt.hist(y,density=True)
    plt.title("Model "+code+" P(Y)")
    plt.show()
    plt.figure()
    plt.hist2d(x,y)
    plt.title("Model " + code + " P(x,y)")
    plt.show()
    [X,Y]=interModel(model,Nx,Ny,2,'x')
    plt.figure()
    plt.hist(Y, density=True)
    plt.title("Model " + code + " P(y) Do(X):=2")
    plt.show()
    [X, Y] = interModel(model, Nx, Ny,2,'y')
    plt.figure()
    plt.hist(X, density=True)
    plt.title("Model " + code + " P(x) Do(Y):=2")
    plt.show()
    plotCond(x, y, code, 0, 'x')
    plotCond(x, y, code, 3, 'x')
    plotCond(x, y, code, -3, 'x')
    plotCond(y, x, code, 0, 'y')
    plotCond(y, x, code, 3, 'y')
    plotCond(y, x, code, -3, 'y')

def genPlots(Nx,Ny,code,type):
    if code=='A':
        [X, Y] = genModel('A', Nx, Ny)
        findDist(X, Y,type,Nx,Ny,'A')
    if code == 'B':
        [X, Y] = genModel('B', Nx, Ny)
        findDist(X, Y,type,Nx,Ny,'B')
def causality(X,Y,type=[]):
    ba=0
    fo=0
    X = np.array(X)
    Y = np.array(Y)
    for levels in range(20):
        # Y->X
        #clf1 = SVR()
        #clf1.fit(Y.reshape(-1, 1), X)
        f = np.poly1d(np.polyfit(Y,X,4))
        residue_back = X - f(Y)
        #residue_back = X - clf1.predict(Y.reshape(-1, 1))
        stat, pvalue_b = Hsic().test(residue_back, Y,1000)
        # X->Y
        #clf2 = SVR()
        #clf2.fit(X.reshape(-1, 1), Y)
        f = np.poly1d(np.polyfit(X, Y, 4))
        residue_for = Y - f(X)
        #residue_for = clf2.predict(X.reshape(-1, 1)) - Y
        stat, pvalue_f = Hsic().test(residue_for, X,1000)
        if pvalue_b<pvalue_f:
            ba+=1
        else:
            fo+=1
    if(ba>=fo):
        print("Final Result for type "+str(type)+":")
        print("X<-Y is AntiCausal")
    else:
        print("Final Result for type " + str(type) + ":")
        print("X->Y is AntiCausal")
    print([pvalue_f,pvalue_b])

def tuebingen(number):
    if number<10:
        with open("pair000"+str(number)+".txt", 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split() for line in stripped if line)
            with open("pair000"+str(number)+".csv", 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerows(lines)
        data = pd.read_csv("pair000"+str(number)+".csv",header=None)
        df = pd.DataFrame(data)
    else:
        with open('pair00'+str(number)+'.txt', 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split() for line in stripped if line)
            with open('pair00'+str(number)+'.csv', 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerows(lines)
        data = pd.read_csv('pair00'+str(number)+'.csv',header=None)
        df = pd.DataFrame(data)
    causality(df[0], df[1],number)
    plt.figure()
    plt.scatter(df[0], df[1])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('scatter'+str(number))
    plt.show()
    if number < 10:
        os.remove('pair000'+str(number)+'.csv')
    else:
        os.remove('pair00'+str(number)+'.csv')
# Question One

size = 100000
#1
Nx111111 = sn(size)
Ny = sn(size)
genPlots(Nx,Ny,'A',1)
#2
Nx = sn(size)
Ny = su(size)
genPlots(Nx,Ny,'B',2)
#3
Nx = su(size)
Ny = sn(size)
genPlots(Nx,Ny,'B',3)
#4
Nx = sn(size)
Ny = ts(size,1)
genPlots(Nx,Ny,'B',4)
#5
Nx = sn(size)
Ny = ts(size,20)
genPlots(Nx,Ny,'B',5)

### Question Two

size = 200
#1
Nx = sn(size)
Ny = sn(size)
[X,Y] = genModel('A', Nx, Ny)
plt.figure()
plt.scatter(X,Y)
plt.show()
causality(X,Y,1)

#2
Nx = sn(size)
Ny = su(size)
[X,Y] = genModel('B', Nx, Ny)
causality(X,Y,2)

#3
Nx = su(size)
Ny = sn(size)
[X,Y] = genModel('B', Nx, Ny)
causality(X,Y,3)

#4
Nx = sn(size)
Ny = ts(size,1)
[X,Y] = genModel('B', Nx, Ny)
causality(X,Y,4)

#5
Nx = sn(size)
Ny = ts(size,20)
[X,Y] = genModel('B', Nx, Ny)
causality(X,Y,5)

#Real Data 1
#eruptions waiting

with open('data.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split() for line in stripped if line)
    with open('data.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)
data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
causality(df['eruptions'],df['waiting'])
plt.figure()
plt.scatter( df['eruptions'],df['waiting'])
plt.ylabel('Waiting')
plt.xlabel('Eruption Time')
plt.title('Old Faithful Geyser Data')
plt.show()

#Real Data 2


numbers = np.arange(15,29)
for i in numbers:
    tuebingen(i)

### Question 3
with open('Data3.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split() for line in stripped if line)
    with open('Data3.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)
data = pd.read_csv("Data3.csv",header=None)
df = pd.DataFrame(data)
plt.figure()
plt.scatter(df[1],df[2])
plt.ylabel('y')
plt.xlabel('x')
plt.title('Q3')
plt.figure()
plt.scatter(df[3],df[4])
plt.ylabel('w')
plt.xlabel('z')
plt.title('Q3')
plt.show()

plt.figure()
x,y = prob3(df[1],df[3],df[2],df[4],0,0)
plt.hist2d(x, y)
plt.show()

plt.figure()
plt.subplot(131)
plt.suptitle('x=0,z=0')
x,y = prob3(df[2],df[4],df[1],df[3],0,0)
plt.hist2d(x, y)
causality(x,y)
plt.xlabel('y')
plt.ylabel('w')
plt.subplot(132)
plt.suptitle('x=-1,z=1')
x,y = prob3(df[2],df[4],df[1],df[3],1,-1)
causality(x,y)
plt.hist2d(x, y)
plt.xlabel('y')
plt.ylabel('w')
plt.subplot(133)
plt.suptitle('x=1,z=-1')
x,y = prob3(df[2],df[4],df[1],df[3],-1,1)
causality(x,y)

plt.hist2d(x, y)
plt.xlabel('y')
plt.ylabel('w')
plt.title('Q3-1')

plt.show()

plt.figure()
plt.subplot(131)
plt.suptitle('w=0,z=0')
x,y = prob3(df[1],df[2],df[3],df[4],0,0)
plt.hist2d(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(132)
plt.suptitle('w=-1,z=1')
x,y = prob3(df[1],df[2],df[3],df[4],1,-1)
plt.hist2d(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(133)
plt.suptitle('w=1,z=-1')
x,y = prob3(df[1],df[2],df[3],df[4],-1,1)
plt.hist2d(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Q3-2')

plt.show()

plt.figure()
plt.subplot(131)
plt.suptitle('w=0,z=0')
x,y = prob3(df[1],df[2],df[3],df[4],0,0)
plt.hist2d(y, x)
plt.xlabel('y')
plt.ylabel('x')
plt.subplot(132)
plt.suptitle('w=-1,z=1')
x,y = prob3(df[1],df[2],df[3],df[4],1,0)
plt.hist2d(y, x)
plt.xlabel('y')
plt.ylabel('x')
plt.subplot(133)
plt.suptitle('w=0,z=par')
x,y = prob3(df[1],df[2],df[3],df[4],-1,0)
plt.hist2d(y, x)
plt.xlabel('y')
plt.ylabel('x')
plt.title('Q3-3')
plt.show()

plt.figure()
plt.subplot(131)
plt.suptitle('w=0,z=0')
x,y = prob3(df[1],df[2],df[3],df[4],0,0)
plt.hist2d(y, x)
plt.xlabel('y')
plt.ylabel('x')
plt.subplot(132)
plt.suptitle('w=-1,z=1')
x,y = prob3(df[1],df[2],df[3],df[4],0,-1)
plt.hist2d(y, x)
plt.xlabel('y')
plt.ylabel('x')
plt.subplot(133)
plt.suptitle('w=par,z=0')
x,y = prob3(df[1],df[2],df[3],df[4],0,1)
plt.hist2d(y, x)
plt.xlabel('y')
plt.ylabel('x')
plt.title('Q3-4')
plt.show()

model = SVR()
arraye = np.array(df[2])
model.fit(arraye.reshape(-1,1), df[1])
residue = model.predict(arraye.reshape(-1,1)) - np.array(df[1])
print(residue)
plt.figure()
plt.title('x = f(y)')
plt.scatter(residue, df[2])
plt.xlabel('Y')
plt.ylabel('Residue')
plt.show()

model = SVR()
arraye = np.array(df[1])
model.fit(arraye.reshape(-1,1), df[2])
residue = model.predict(arraye.reshape(-1,1)) - np.array(df[2])
plt.figure()
plt.title('y = f(x)')
plt.scatter(residue, df[1])
plt.xlabel('X')
plt.ylabel('Residue')
plt.show()

model = SVR()
model.fit(df[[1,2]], df[3])
residue = model.predict(df[[1,2]]) - df[3]
plt.figure()
plt.suptitle('z = f(x,y)')
plt.subplot(121)
plt.scatter(residue, df[1])
plt.xlabel('X')
plt.ylabel('Residue')
plt.subplot(122)
plt.scatter(residue, df[2])
plt.xlabel('Y')
plt.ylabel('Residue')
plt.show()

model = SVR()
arraye = np.array(df[2])
model.fit(arraye.reshape(-1,1), df[4])
residue = model.predict(arraye.reshape(-1,1)) - df[4]
plt.figure()
plt.title('w = f(y)')
plt.scatter(residue, df[2])
plt.xlabel('y')
plt.ylabel('Residue')
plt.show()

model = SVR()
arraye = np.array(df[2])
model.fit(arraye.reshape(-1,1), df[3])
residue = model.predict(arraye.reshape(-1,1)) - df[3]
plt.figure()
plt.title('z = f(y)')
plt.scatter(residue, df[2])
plt.xlabel('y')
plt.ylabel('Residue')
plt.show()


model = SVR()
model.fit(df[[2,3,4]], df[1])
residue = model.predict(df[[2,3,4]]) - df[1]
plt.figure()
plt.subplot(131)
plt.suptitle('x = f(y,w,z)')
plt.scatter(residue, df[2])
plt.xlabel('Y')
plt.ylabel('Residue')
plt.subplot(132)
plt.scatter(residue, df[3])
plt.xlabel('Z')
plt.ylabel('Residue')
plt.subplot(133)
plt.scatter(residue, df[4])
plt.xlabel('W')
plt.ylabel('Residue')
plt.show()


model = SVR()
model.fit(df[[3,4]], df[1])
residue = model.predict(df[[3,4]]) - df[1]
plt.figure()
plt.suptitle('x = f(w,z)')
plt.subplot(121)
plt.scatter(residue, df[3])
plt.xlabel('Z')
plt.ylabel('Residue')
plt.subplot(122)
plt.scatter(residue, df[4])
plt.xlabel('W')
plt.ylabel('Residue')
plt.show()

model = SVR()
model.fit(df[[2,4]], df[1])
residue = model.predict(df[[2,4]]) - df[1]
plt.figure()
plt.suptitle('x = f(w,y)')
plt.subplot(121)
plt.scatter(residue, df[2])
plt.xlabel('Y')
plt.ylabel('Residue')
plt.subplot(122)
plt.scatter(residue, df[4])
plt.xlabel('W')
plt.ylabel('Residue')
plt.show()

model = SVR()
model.fit(df[[3,2]], df[1])
residue = model.predict(df[[3,2]]) - df[1]
plt.figure()
plt.suptitle('x = f(z,y)')
plt.subplot(121)
plt.scatter(residue, df[3])
plt.xlabel('Z')
plt.ylabel('Residue')
plt.subplot(122)
plt.scatter(residue, df[2])
plt.xlabel('Y')
plt.ylabel('Residue')
plt.show()


model = SVR()
arraye = np.array(df[4])
model.fit(arraye.reshape(-1,1), df[1])
residue = model.predict(arraye.reshape(-1,1)) - df[1]
plt.figure()
plt.title('x = f(w)')
plt.scatter(residue, df[4])
plt.xlabel('W')
plt.ylabel('Residue')
plt.show()

model = SVR()
arraye = np.array(df[3])
model.fit(arraye.reshape(-1,1), df[1])
residue = model.predict(arraye.reshape(-1,1)) - df[1]
plt.figure()
plt.title('x = f(z)')
plt.scatter(residue, df[3])
plt.xlabel('Z')
plt.ylabel('Residue')
plt.show()


model = SVR()
model.fit(df[[1,3,4]], df[2])
residue = model.predict(df[[1,3,4]]) - df[2]
plt.figure()
plt.subplot(131)
plt.suptitle('y = f(x,w,z)')
plt.scatter(residue, df[1])
plt.xlabel('X')
plt.ylabel('Residue')
plt.subplot(132)
plt.scatter(residue, df[3])
plt.xlabel('Z')
plt.ylabel('Residue')
plt.subplot(133)
plt.scatter(residue, df[4])
plt.xlabel('W')
plt.ylabel('Residue')
plt.show()

model = SVR()
model.fit(df[[3,4]], df[2])
residue = model.predict(df[[3,4]]) - df[2]
plt.figure()
plt.suptitle('y = f(w,z)')
plt.subplot(121)
plt.scatter(residue, df[3])
plt.xlabel('Z')
plt.ylabel('Residue')
plt.subplot(122)
plt.scatter(residue, df[4])
plt.xlabel('W')
plt.ylabel('Residue')
plt.show()

model = SVR()
model.fit(df[[1,3]], df[2])
residue = model.predict(df[[1,3]]) - df[2]
plt.figure()
plt.suptitle('y = f(x,z)')
plt.subplot(121)
plt.scatter(residue, df[3])
plt.xlabel('Z')
plt.ylabel('Residue')
plt.subplot(122)
plt.scatter(residue, df[1])
plt.xlabel('X')
plt.ylabel('Residue')
plt.show()

model = SVR()
model.fit(df[[1,4]], df[2])
residue = model.predict(df[[1,4]]) - df[2]
plt.figure()
plt.suptitle('y = f(x,w)')
plt.subplot(121)
plt.scatter(residue, df[4])
plt.xlabel('W')
plt.ylabel('Residue')
plt.subplot(122)
plt.scatter(residue, df[1])
plt.xlabel('X')
plt.ylabel('Residue')
plt.show()



model = SVR()
arraye = np.array(df[4])
model.fit(arraye.reshape(-1,1), df[2])
residue = model.predict(arraye.reshape(-1,1)) - df[2]
plt.figure()
plt.title('y = f(w)')
plt.scatter(residue, df[4])
plt.xlabel('w')
plt.ylabel('Residue')
plt.show()

model = SVR()
arraye = np.array(df[3])
model.fit(arraye.reshape(-1,1), df[2])
residue = model.predict(arraye.reshape(-1,1)) - df[2]
plt.figure()
plt.title('y = f(z)')
plt.scatter(residue, df[3])
plt.xlabel('z')
plt.ylabel('Residue')
plt.show()

## Y = f(x,z,w) X = f(w,z)
