#Alumno: Ortiz Barajas Jesús Germán
#Aprendizaje 2018-2
#Regla de aprendizaje del perceptrón

#Ejercicio 3

import numpy as np

def obtainN(weight, p, bias):
    n=(weight*p)+bias
    return n

def hardlim(n):
    if(n<0):
        return 0
    else:
        return 1

#Se transforma la función hardlim para que pueda ser aplicada a los vectores
matrixHardlim= np.vectorize(hardlim)

def obtainA(n):
    a = matrixHardlim(n)
    return a

def error(target, a):
    error = target - a
    return error

def updateWeight(weight, error, p):
    weight = weight + (error*p.transpose())
    return weight

def updateBias(bias, error):
    bias = bias + error
    return bias

def setParameters(categories, weight, bias):
    w=weight
    b= bias
    idealError=np.matrix('0;0')
    boolError=False
    indexCategories=0
    auxCategory=np.matrix('0;0')
    while (boolError!=True):
        n = obtainN(w,categories[indexCategories][0],b)
        a = obtainA(n)
        e = error(categories[indexCategories][1], a)
        if((e==idealError).all()==False):
            w = updateWeight(w,e,categories[indexCategories][0])
            b = updateBias(b,e)
            auxCategory=categories[indexCategories][0]
        if(indexCategories==7):
            indexCategories=0
        else:
            indexCategories=indexCategories+1
        if(((e==idealError).all()==True) and ((categories[indexCategories][0]==auxCategory).all())==True):
            boolError=True
            parameters=[w, b]
    return parameters


#Se definen las categorias a clasificar
p1 = np.matrix('1;1')
t1 = np.matrix('0;0')
cat1 = [p1,t1]

p2 = np.matrix('1;2')
t2 = np.matrix('0;0')
cat2 = [p2,t2]

p3 = np.matrix('2;-1')
t3 = np.matrix('0;1')
cat3 = [p3,t3]

p4 = np.matrix('2;0')
t4 = np.matrix('0;1')
cat4 = [p4,t4]

p5 = np.matrix('-1;2')
t5 = np.matrix('1;0')
cat5 = [p5,t5]

p6 = np.matrix('-2;1')
t6 = np.matrix('1;0')
cat6 = [p6,t6]

p7 = np.matrix('-1;-1')
t7 = np.matrix('1;1')
cat7 = [p7,t7]

p8 = np.matrix('-2;-2')
t8 = np.matrix('1;1')
cat8 = [p8,t8]

categories = [cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8]

#Se definen pesos y bias
w=np.matrix('4 0;0 4')
b=np.matrix('4;4')

result=setParameters(categories, w, b)

print("Peso:")
print(result[0])

print("Bias: ")
print(result[1])
