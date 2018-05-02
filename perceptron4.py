#Alumno: Ortiz Barajas Jesús Germán
#Aprendizaje 2018-2
#Regla de aprendizaje del perceptrón

#Ejercicio 4

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
p11 = np.matrix('-1;1')
p12 = np.matrix('-1;0')
t1 = np.matrix('0;1')
cat11 = [p11, t1]
cat12 = [p12, t1]

p21 = np.matrix('0;2')
p22 = np.matrix('1;2')
t2 = np.matrix('1;1')
cat21 = [p21, t2]
cat22 = [p22, t2]

p31 = np.matrix('2;0')
p32 = np.matrix('2;1')
t3 = np.matrix('1;0')
cat31 = [p31, t3]
cat32 = [p32, t3]

p41 = np.matrix('1;-1')
p42 = np.matrix('0;-1')
t4 = np.matrix('0;0')
cat41 = [p41, t4]
cat42 = [p42, t4]

categories = [cat11, cat12, cat21, cat22, cat31, cat32, cat41, cat42]

#Definimos pesos y bias
w = np.matrix('3 -3;-3 3')
b = np.matrix('-3;3')

result=setParameters(categories, w, b)

print("Peso:")
print(result[0])

print("Bias: ")
print(result[1])

