import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to a valid option like 'TkAgg'
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import time
import pandas as pd

def abrirImagenesEscaladas( carpeta, escala=32 ):
    # abre todas las imagenes de la carpeta, y las escala de tal forma que midan (escala x escala)px
    # devuelve las imagenes aplanadas -> vectores de tamano escala^2 con valores entre 0 y 1
    imagenes = []

    for dirpath, dirnames, filenames in os.walk(carpeta):
        for file in filenames:
            if file.endswith('DS_Store'):
                continue
            img = Image.open( os.path.join(carpeta, file) )
            img = img.resize((escala, escala))
            img.convert('1')
            img = np.asarray(img)
            if len(img.shape)==3:
                img = img[:,:,0].reshape((escala**2 )) / 255
            else:
                img = img.reshape((escala**2 )) / 255
            
            imagenes.append( img )

    return imagenes

def balancear_datos(imagenes_entrenamiento):

    img_train_sin_neumonia = imagenes_entrenamiento[0]
    img_train_neumonia =imagenes_entrenamiento[1]
    img_test_sin_neumonia =imagenes_entrenamiento[2]
    img_test_neumonia = imagenes_entrenamiento[3]

    # MAX NUMBER OF IMAGES
    n_train = min(len(img_train_sin_neumonia), len(img_train_neumonia))
    n_test = min(len(img_test_sin_neumonia), len(img_test_neumonia))

    # BALANCE
    img_train_sin_neumonia = img_train_sin_neumonia[:n_train]
    img_train_neumonia = img_train_neumonia[:n_train]
    img_test_sin_neumonia = img_test_sin_neumonia[:n_test]
    img_test_neumonia = img_test_neumonia[:n_test]

    
    return (img_train_sin_neumonia, img_train_neumonia, img_test_sin_neumonia, img_test_neumonia)

## Ejercicio 1

def F(i,w,b):
    """_summary_

    Args:
        i (Vector): imagen reshaped a un vector de tamano 32^2
        w (Vector): Pesos de la red
        b (Float): Bias de la red

    Returns:
        probabolidad: 0 < p < 1: Probabilidad de que la imagen sea un 1 (Tiene neumonia)
    """
    tan = np.tanh(np.dot(w,i)+b)
    return (tan + 1)/2

# Derivada de L con respecto a W
def L_w(i,w,b,d):
    """_summary_

    Args:
        i (Vector): imagen reshaped a un vector de tamano 32^2
        w (Vector): Pesos de la red
        b (Float): Bias de la red

    Returns:
        Vector: Gradiente de la probabilidad con respecto a los pesos
    """
    # t0=tanh(b+W⊤⋅i)
    #return: (1−t0^2)⋅((1+t0)/2−d)⋅i
    
    t0 = np.tanh(np.dot(w,i)+b)
    return (1-t0**2)*(((1+t0)/2)-d) * i

# Derivada de L con respecto a b
def L_b(i,w,b,d):
    """_summary_

    Args:
        i (Vector): imagen reshaped a un vector de tamano 32^2
        w (Vector): Pesos de la red
        b (Float): Bias de la red

    Returns:
        Float: Gradiente de la probabilidad con respecto al bias
    """
    #t0=tanh(b+W⊤⋅i)
    #(1−t0^2)⋅((1+t0)/2−d)   
    t0 = np.tanh(np.dot(w,i)+b)
    return (1-t0**2)*(((1+t0)/2)-d)
## Ejercicio 2

def desenso_gradiente(w,b,gradiente_w, gradiente_b, d, alpha=0.1):
    """_summary_

    Args:
        imagenes_entrenamiento (List): Lista de imagenes de tamano 32^2
        w (Vector): Pesos de la red
        b (Float): Bias de la red
        alpha (Float): Learning rate

    Returns:
        Tuple: Pesos y bias actualizados

    """
    w = w - alpha * gradiente_w
    b = b - alpha * gradiente_b
    return w,b


## Ejercicio 3



def train(datos, alpha=0.005, epochs = 5,seed = 42,plot_graph=True):
    """_summary_

    Args:
        datos (tuple): Tupla de dos listas, la primera con las imagenes Normales y la segunda con las imagenes con Neumonia 
        w (Vector): Pesos de la red
        b (Float): Bias de la red
        alpha (Float): Learning rate
        epochs (Int): Numero de iteraciones

    Returns:
        Tuple: Pesos y bias actualizados
    """
    # inicioamos con pesos aleatorios
    # set numpy seed
    

    datos_sin_neumonia = datos[0]
    datos_con_neumonia = datos[1]

    
    np.random.seed(seed)
    w = np.random.randn(datos_sin_neumonia[0].shape[0])
    b = np.random.randn(1)
    #  Separar datos en entrenamiento y validación
     
    # datos_sin_neumonia = np.array(datos[0])
    # datos_con_neumonia = np.array(datos[1])

    # X = np.vstack((datos_sin_neumonia, datos_con_neumonia))
    # y = np.hstack((np.zeros(len(datos_sin_neumonia)), np.ones(len(datos_con_neumonia))))

    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # np.random.seed(seed)
    # w = np.random.randn(X_train.shape[1])
    # b = np.random.randn(1)
    
    print(len(datos_sin_neumonia)+len(datos_con_neumonia),len(datos_sin_neumonia)/len(datos_con_neumonia))
    
   
    errores = []
    errores_train = []
    
    errores_val = []
    
    for epoch in range(epochs):  
        
        error = 0
        gradiente_w = np.zeros_like(w)
        gradiente_b = np.zeros_like(b)
        
        # for xi, yi in zip(X_train, y_train):
        #     gradiente_w += L_w(xi, w, b, yi)
        #     gradiente_b += L_b(xi, w, b, yi)
        #     error += (F(xi, w, b) - yi)**2
            
            
        # # Actualización de los parámetros
        # w, b = desenso_gradiente(w, b, gradiente_w, gradiente_b, alpha)

        # # Calcular error en el conjunto de validación
        # for xi_val, yi_val in zip(X_val, y_val):
        #     errores += (F(xi_val, w, b) - yi_val)**2

        # # Almacenar los errores promedios
        # errores_train.append(error / len(X_train))
        # errores_val.append(errores / len(X_val))

        # # Decaer la tasa de aprendizaje
        # alpha *= 0.95

        # # Mostrar el error de la epoch actual
        # if epoch % 50 == 0 or epoch == epochs - 1:
        #     print(f"Epoch {epoch}: Error de Entrenamiento = {errores_train[-1]f}, Error de Validación = {errores_val[-1]:.4f}")
        
        # Entrenamiento con imágenes sin neumonía
        label = 0
        for i in datos_sin_neumonia: 
            gradiente_w += L_w(i,w,b,label)
            gradiente_b += L_b(i,w,b,label)
            error += (F(i,w,b))**2 # Falta arreglar esto
            
        # w,b = desenso_gradiente(w,b,gradiente_w,gradiente_b,label,alpha)
            
        # Entrenamiento con imágenes con neumonía
        label = 1
        for i in datos_con_neumonia:
            gradiente_w += L_w(i,w,b,label)
            gradiente_b += L_b(i,w,b,label)
            error += (F(i,w,b) - label)**2 # Falta arreglar esto
        
        # # Promediar los gradientes acumulados
        # gradiente_w /= (len(datos_sin_neumonia) + len(datos_con_neumonia))
        # gradiente_b /= (len(datos_sin_neumonia) + len(datos_con_neumonia))
        
        # Actualización de los parámetros
        w, b = desenso_gradiente(w, b, gradiente_w, gradiente_b, alpha)
        
        # # Almacenar el error cuadrático promedio para visualización
        # errores.append(error / (len(datos_sin_neumonia) + len(datos_con_neumonia)))
        
        # Almacenar el error cuadrático  para visualización
        errores.append(error)
       
        
        # Decaer la tasa de aprendizaje
        alpha *= 0.95
        
        
        # Mostrar el error de la epoch actual
        print(f"\r{error / (len(datos_sin_neumonia) + len(datos_con_neumonia))}",end='',)
    
    if plot_graph:
        plt.plot(errores)
        plt.xlabel('Epoch')
        plt.ylabel('Error Cuadrático')
        plt.title('Error Cuadrático durante el Entrenamiento')
        plt.show()

    return w,b

def test(w,b,datos):
    """_summary_

    Args:
        w (Vector): Pesos de la red
        b (Float): Bias de la red
        datos (tuple): Tupla de dos listas, la primera con las imagenes Normales y la segunda con las imagenes con Neumonia 

    Returns:
        Float: Accuracy
    """
    datos_sin_neumonia = datos[0]
    datos_con_neumonia = datos[1]
    correctos = 0
    for i in datos_sin_neumonia:
        if F(i,w,b) < 0.5:
            correctos += 1
    for i in datos_con_neumonia:
        if F(i,w,b) >= 0.5:
            correctos += 1
    return correctos/(len(datos_sin_neumonia)+len(datos_con_neumonia))


## Ejercicio 4

def analizar_convergencia(errores):
    if len(errores) < 2:
        return float('inf') 
    return abs(errores[-1] - errores[-2]) / errores[-2]

def train_test_convergencia(datos, alpha,seed,plot_graph=True):
    """_summary_

    Args:
        datos (tuple): Tupla de dos listas, la primera con las imagenes Normales y la segunda con las imagenes con Neumonia 
        w (Vector): Pesos de la red
        b (Float): Bias de la red
        alpha (Float): Learning rate
        epochs (Int): Numero de iteraciones

    Returns:
        Tuple: Pesos y bias actualizados
    """
    # inicioamos con pesos aleatorios
    # set numpy seed
    

    datos_sin_neumonia = datos[0]
    datos_con_neumonia = datos[1]
    

    np.random.seed(seed)
    w = np.random.randn(datos_sin_neumonia[0].shape[0])
    b = np.random.randn(1)
    
    
    print(len(datos_sin_neumonia)+len(datos_con_neumonia),len(datos_sin_neumonia)/len(datos_con_neumonia))
    
    error = 0
    errores = []
    contador = 0
    while(True):  
        error = 0
        gradiente_w = np.zeros_like(w)
        gradiente_b = np.zeros_like(b)
        
        # Entrenamiento con imágenes sin neumonía
        label = 0
        for i in datos_sin_neumonia: 
            
            gradiente_w += L_w(i,w,b,label)
            gradiente_b += L_b(i,w,b,label)
            error += (F(i,w,b) - label)**2 # Falta arreglar esto
            
        # w,b = desenso_gradiente(w,b,gradiente_w,gradiente_b,label,alpha)
            
        # Entrenamiento con imágenes con neumonía
        label = 1
        for i in datos_con_neumonia:
            gradiente_w += L_w(i,w,b,label)
            gradiente_b += L_b(i,w,b,label)
            error += (F(i,w,b) - label)**2 # Falta arreglar esto
        
       
        w, b = desenso_gradiente(w, b, gradiente_w, gradiente_b, alpha)
        
        errores.append(error / (len(datos_sin_neumonia) + len(datos_con_neumonia)))
              
    
        # Decaer la tasa de aprendizaje
        alpha *= 0.95
        
        diferencia = analizar_convergencia(errores)
        
        # Mostrar el error de la epoch actual
        print(f"\rError: {errores[-1]}  Diferencia: {diferencia}",end='',)
        
        if diferencia < 1e-11:
            print(f"Convergio en el epoch: {contador} con aplha: {alpha}")
            break
        
        contador += 1
       
        
    if plot_graph:
        plt.plot(errores)
        plt.xlabel('Epoch')
        plt.ylabel('Error Cuadrático')
        plt.title('Error Cuadrático durante el Entrenamiento')
        plt.show()

    return w,b,contador,alpha


def train(datos, alpha=0.005, epochs = 5,seed = 42,plot_graph=True):
    """_summary_

    Args:
        datos (tuple): Tupla de dos listas, la primera con las imagenes Normales y la segunda con las imagenes con Neumonia 
        w (Vector): Pesos de la red
        b (Float): Bias de la red
        alpha (Float): Learning rate
        epochs (Int): Numero de iteraciones

    Returns:
        Tuple: Pesos y bias actualizados
    """
    # inicioamos con pesos aleatorios
    # set numpy seed
    

    datos_sin_neumonia = datos[0]
    datos_con_neumonia = datos[1]

    
    np.random.seed(seed)
    w = np.random.randn(datos_sin_neumonia[0].shape[0])
    b = np.random.randn(1)
    #  Separar datos en entrenamiento y validación
     
    # datos_sin_neumonia = np.array(datos[0])
    # datos_con_neumonia = np.array(datos[1])

    # X = np.vstack((datos_sin_neumonia, datos_con_neumonia))
    # y = np.hstack((np.zeros(len(datos_sin_neumonia)), np.ones(len(datos_con_neumonia))))

    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # np.random.seed(seed)
    # w = np.random.randn(X_train.shape[1])
    # b = np.random.randn(1)
    
    print(len(datos_sin_neumonia)+len(datos_con_neumonia),len(datos_sin_neumonia)/len(datos_con_neumonia))
    
   
    errores = []
    errores_train = []
    
    errores_val = []
    
    for epoch in range(epochs):  
        
        error = 0
        gradiente_w = np.zeros_like(w)
        gradiente_b = np.zeros_like(b)
        
        # for xi, yi in zip(X_train, y_train):
        #     gradiente_w += L_w(xi, w, b, yi)
        #     gradiente_b += L_b(xi, w, b, yi)
        #     error += (F(xi, w, b) - yi)**2
            
            
        # # Actualización de los parámetros
        # w, b = desenso_gradiente(w, b, gradiente_w, gradiente_b, alpha)

        # # Calcular error en el conjunto de validación
        # for xi_val, yi_val in zip(X_val, y_val):
        #     errores += (F(xi_val, w, b) - yi_val)**2

        # # Almacenar los errores promedios
        # errores_train.append(error / len(X_train))
        # errores_val.append(errores / len(X_val))

        # # Decaer la tasa de aprendizaje
        # alpha *= 0.95

        # # Mostrar el error de la epoch actual
        # if epoch % 50 == 0 or epoch == epochs - 1:
        #     print(f"Epoch {epoch}: Error de Entrenamiento = {errores_train[-1]f}, Error de Validación = {errores_val[-1]:.4f}")
        
        # Entrenamiento con imágenes sin neumonía
        label = 0
        for i in datos_sin_neumonia: 
            gradiente_w += L_w(i,w,b,label)
            gradiente_b += L_b(i,w,b,label)
            error += (F(i,w,b) - label)**2 # Falta arreglar esto
            
        # w,b = desenso_gradiente(w,b,gradiente_w,gradiente_b,label,alpha)
            
        # Entrenamiento con imágenes con neumonía
        label = 1
        for i in datos_con_neumonia:
            gradiente_w += L_w(i,w,b,label)
            gradiente_b += L_b(i,w,b,label)
            error += (F(i,w,b) - label)**2 # Falta arreglar esto
            
       
        
        # # Promediar los gradientes acumulados
        # gradiente_w /= (len(datos_sin_neumonia) + len(datos_con_neumonia))
        # gradiente_b /= (len(datos_sin_neumonia) + len(datos_con_neumonia))
        
        # Actualización de los parámetros
        w, b = desenso_gradiente(w, b, gradiente_w, gradiente_b, alpha)
        
        
        # Almacenar el error cuadrático promedio para visualización
        # errores.append(error / (len(datos_sin_neumonia) + len(datos_con_neumonia)))
        
        # Almacenar el error cuadrático  para visualización
        errores.append(error)
  
        # Decaer la tasa de aprendizaje
        alpha *= 0.95
        
        
        # Mostrar el error de la epoch actual
        print(f"\r{error / (len(datos_sin_neumonia) + len(datos_con_neumonia))}",end='',)
    
    if plot_graph:
        plt.plot(errores)
        plt.xlabel('Epoch')
        plt.ylabel('Error Cuadrático')
        plt.title('Error Cuadrático durante el Entrenamiento')
        plt.show()

    return w,b

def main():

    # Data
    if(True):
        img_train_sin_neumonia = abrirImagenesEscaladas('./chest_xray/train/NORMAL/')
        img_train_neumonia = abrirImagenesEscaladas('./chest_xray/train/PNEUMONIA/') # NO FUNCIONA :(
        img_test_sin_neumonia = abrirImagenesEscaladas('./chest_xray/test/NORMAL/')
        img_test_neumonia = abrirImagenesEscaladas('./chest_xray/test/PNEUMONIA/')

        data = (img_train_sin_neumonia, img_train_neumonia, img_test_sin_neumonia, img_test_neumonia)

    data = balancear_datos(data)

    train_sin, train_con, test_sin, test_con = data
    training_data = (train_sin,train_con)
    testing_data = (test_sin,test_con)

    w_res, b_res = train(
        training_data,
        alpha=0.001,
        epochs = 500,
        seed = 40,
        plot_graph=True
        )

    test(w_res,b_res,training_data)


    alphas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    tiempo_ejecucion = []
    epochs = []
    alphas_nuevos_valores = []
    porcentaje_correctitud = []

    for i in alphas:
        
        start_time = time.time()
        
        w_res, b_res, epoch, alpha = train_test_convergencia(
            (train_sin, train_con),
            alpha=i,
            seed = 42,
            plot_graph=False
            )
        
        end_time = time.time()
        
        tiempo_ejecucion.append(end_time - start_time)
        epochs.append(epoch)
        alphas_nuevos_valores.append(alpha)
        
        porcentaje_correctitud.append(test(w_res,b_res,(test_sin,test_con)))
        


    # Crear un DataFrame con los datos

    # data = {
    #     'Metric': ['tiempo_ejecucion', 'alphas_nuevos_valores'],
    #     **{alpha: [tiempo_ejecucion[i], alphas_nuevos_valores[i]] for i, alpha in enumerate(alphas)}
    # }


    # df = pd.DataFrame(data)

    # # Nombre del archivo con el valor de convergencia en el título
    # filename = f'results_convergence_{1e-11:.0e}.csv'

    # # Guardar el DataFrame en un archivo CSV
    # df.to_csv(filename, index=False)

    # print(f'Data saved to {filename}')

    plt.figure(figsize=(10, 6))

    alphas_str = [str(alpha) for alpha in alphas]

    plt.bar(alphas_str, tiempo_ejecucion, color='skyblue')

    plt.xlabel('Valores de aplha')
    plt.ylabel('Tiempo de ejecucion (segundos)')
    plt.title('Tiempos de ejecucion para cada alpha')

    plt.show()

    plt.figure(figsize=(10, 6))

    alphas_str = [str(alpha) for alpha in alphas]
    plt.bar(alphas_str, alphas_nuevos_valores, color='skyblue')

    plt.xlabel('Valores de aplha')
    plt.ylabel('Valores finales de alpha')
    plt.title('Progresion del alpha segun el punto de inicio')

    plt.show()

    plt.plot(alphas_str, alphas_nuevos_valores, marker='o', linestyle='-', color='b')  # Gráfico de líneas

    plt.xlabel('Valores de aplha')
    plt.ylabel('Valores finales de alpha')
    plt.title('Progresion del alpha segun el punto de inicio')

    plt.grid(True) 
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.bar(alphas_str, epochs, color='skyblue')

    plt.xlabel('Valores de aplha')
    plt.ylabel('Iteraciones')
    plt.title('Cantidad de iteraciones hasta convergencia')

    plt.show()
    ## Ejercicio 5

    # # Data
    # esclados = [32,64,128]
    # tiempo_ejecucion_esclado = []
    # efectividad = []


    # for i in esclados:
    #     img_train_sin_neumonia = abrirImagenesEscaladas('./chest_xray/train/NORMAL/',i)
    #     img_train_neumonia = abrirImagenesEscaladas('./chest_xray/train/PNEUMONIA/',i) # NO FUNCIONA :(
    #     img_test_sin_neumonia = abrirImagenesEscaladas('./chest_xray/test/NORMAL/',i)
    #     img_test_neumonia = abrirImagenesEscaladas('./chest_xray/test/PNEUMONIA/',i)

    #     data = (img_train_sin_neumonia, img_train_neumonia, img_test_sin_neumonia, img_test_neumonia)
        
    #     data = balancear_datos(data)
        
    #     start_time = time.time()
    #     w_res, b_res = train(
    #         (train_sin, train_con),
    #         alpha=i,
    #         epochs=1500,
    #         seed = 42
    #         )
    #     end_time = time.time()
    #     tiempo_ejecucion_esclado.append(end_time - start_time)
        
    #     efectividad.append(test(w_res,b_res,(test_sin,test_con), 0.5))
        
        
    # plt.figure(figsize=(10, 6))

    # plt.plot(esclados, tiempo_ejecucion_esclado, marker='o', linestyle='-', color='b', label='Tiempo de ejecucion')  # Gráfico de líneas

    # plt.xlabel('Esclados')
    # plt.ylabel('Tiempo de ejecucion (segundos)')
    # plt.title('Tiempos de ejecucion para cada esclado')
    # plt.legend()  
    # plt.grid(True) 
    # plt.show()
    # plt.figure(figsize=(10, 6))

    # plt.plot(esclados, efectividad, marker='o', linestyle='-', color='b', label='Tiempo de ejecucion')  # Gráfico de líneas

    # plt.xlabel('Esclados')
    # plt.ylabel('Tiempo de ejecucion (segundos)')
    # plt.title('Tiempos de ejecucion para cada esclado')
    # plt.legend()  
    # plt.grid(True) 
    # plt.show()
    
if __name__ == "__main__":
    main()