#Alumno: Ortiz Barajas Jesús Germán
#Aprendizaje 2018-2
#Regla de aprendizaje del perceptrón

#Ejercicio 1. Compuerta AND

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
    idealError=np.matrix('0')
    boolError=False
    indexCategories=0
    auxCategory=np.matrix('0')
    while (boolError!=True):
        n = obtainN(w,categories[indexCategories][0],b)
        a = obtainA(n)
        e = error(categories[indexCategories][1], a)
        if((e==idealError).all()==False):
            w = updateWeight(w,e,categories[indexCategories][0])
            b = updateBias(b,e)
            auxCategory=categories[indexCategories][0]
        if(indexCategories==3):
            indexCategories=0
        else:
            indexCategories=indexCategories+1
        if(((e==idealError).all()==True) and ((categories[indexCategories][0]==auxCategory).all())==True):
            boolError=True
            parameters=[w, b]
    return parameters

#Definimos las categorias a clasificar
p11 = np.matrix('0;0')
p12 = np.matrix('0;1')
p13 = np.matrix('1;0')
t1 = np.matrix('0')

cat11 = [p11, t1]
cat12 = [p12, t1]
cat13 = [p13, t1]

p2 = np.matrix('1;1')
t2 = np.matrix('1')
cat2 = [p2, t2]

categories = [cat11, cat12, cat13, cat2]

w=np.matrix('-7 -5')
b=np.matrix('4')

result=setParameters(categories, w, b)

print("Peso:")
print(result[0])

print("Bias: ")
print(result[1])