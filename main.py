import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import os


def dataF():

    filename = "data.csv"
    nombreColumnas = ["Porcentaje", "Tiempo"]
    dataFrame = pd.read_csv(
        filename, names=nombreColumnas, delimiter=",", header=0)
    dataFrameClean = dataFrame.drop([95, 96, 97, 98, 99])

    return dataFrameClean


def regresionLineal(dataFrame):
    txtAn.delete('1.0', END)
    arrayData = dataFrame.values
    X_reg = arrayData[:, 0:1]
    Y_reg = arrayData[:, 1:2]

    test_size = 0.2
    seed = 8
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_reg, Y_reg, test_size=test_size, random_state=seed)

    model = LinearRegression()
    # Entrenamos el modelo con los datos de training
    model.fit(X_train, Y_train)
    txtAn.insert(END, "=======================\n")
    txtAn.insert(END, f"El intercepto es {model.intercept_}\n")
    txtAn.insert(END, f"El Coeficiente es {model.coef_}\n")

    interc = model.intercept_[0]
    coef = model.coef_[0][0]

    Y_pred = model.predict(X_test)
    txtAn.insert(END, "=======================\n")
    txtAn.insert(END, f"Prediccion \n{Y_pred}\n")
    txtAn.insert(END, f"Lo que debería darnos es \n{Y_test}\n")

    MSE = mean_squared_error(Y_test, Y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(Y_test, Y_pred)

    txtAn.insert(END, "=======================\n")

    txtAn.insert(END, f"MSE es {MSE}\n")
    txtAn.insert(END, f"RMSE es {RMSE}\n")
    txtAn.insert(END, f"R2 es {R2}\n")
    txtAn.insert(END, "=======================\n")

    #plt.scatter(X_train, Y_train, color='b')
    # plt.show()
    #plt.scatter(Y_test, Y_pred, color='g')
    #plt.plot([0,100],[0,100], color='r')
    # plt.show()

    '''
    plt.scatter(X_train, Y_train, color='b')
    X_min = X_train.min()
    X_max = X_train.max()
    Y_min = X_min*coef+interc
    Y_max = X_max*coef+interc
    plt.plot([X_min, X_max], [Y_min, Y_max], color='r')
    '''

    X_min = X_reg.min()
    X_max = X_reg.max()
    Y_min = X_min*coef+interc
    Y_max = X_max*coef+interc

    # plt.show()

    if(interc >= 0):
        y_cad = "y = "+str(coef)+"x +" + str(interc)

    else:
        y_cad = "y = "+str(coef)+"x -" + str(-interc)

    txtAn.insert(END, y_cad+"\n")

    fig = plt.figure(figsize=(5, 4.2))

    plt.title("REGRESION LINEAL")
    plt.scatter(X_reg, Y_reg, color='b', label="Tiempo en ms")
    plt.plot([X_min, X_max], [Y_min, Y_max], color='r', label=y_cad)
    plt.xlabel('PORCENTAJE')
    plt.ylabel('TIEMPO EN MILISEGUNDOS')
    plt.legend()
    plt.grid()
    canvas = FigureCanvasTkAgg(fig, master=root)
    plot_widget = canvas.get_tk_widget()
    plot_widget.place(x=620, y=180)


def clustering(dataFrame):
    txtAn.delete('1.0', END)
    nClust = comboClus.current()+1
    if nClust == 0:
        messagebox.showerror(
            message="INGRESE NUMERO DE PETICIONES", title="ERROR!!!")
    else:

        arr = dataFrame.values
        X = arr[:, 0:1]
        Y = arr[:, 1:2]

        plt.scatter(X, Y, c='g')

        kmeans = KMeans(n_clusters=nClust)
        kmeans.fit(arr)

        txtAn.insert(
            END, f"Los centroides del modelo son: \n{kmeans.cluster_centers_}\n")
        txtAn.insert(END, f"Las etiquetas de cada punto \n{kmeans.labels_}\n")

        i = 1
        for x, y in kmeans.cluster_centers_:
            txtAn.insert(END, "=====================\n")
            txtAn.insert(END, f"Centroide {i}\n")
            txtAn.insert(END, f"x es {x} y es {y}\n")
            txtAn.insert(END, "=====================\n")
            i += 1

        centroids = kmeans.cluster_centers_
        X_centroid = centroids[:, 0]
        Y_centroid = centroids[:, 1]

        plt.scatter(X_centroid, Y_centroid, c='g')
        #plt.scatter(X, Y, c=kmeans.labels_, cmap="rainbow")

        dbscan = DBSCAN(eps=6, min_samples=2)
        dbscan.fit(arr)
        # dbscan.labels_
        plt.scatter(X, Y, c=dbscan.labels_, cmap="rainbow")
        # plt.show()
        # aqui

        n_clus = len(set(dbscan.labels_))-(1 if -1 in dbscan.labels_ else 0)
        num_noise = list(dbscan.labels_).count(-1)
        txtAn.insert(END,
                     f"Numero de clusters {n_clus} Número de puntos de ruido {num_noise}")

        canvas1 = tk.Canvas(root, width=100, height=100)
        # canvas1.pack()

        label1 = tk.Label(root, text=centroids, justify='center')
        canvas1.create_window(70, 50, window=label1)

        figure1 = plt.Figure(figsize=(5, 4), dpi=100)

        ax1 = figure1.add_subplot(1, 1, 1)

        #"Porcentaje", "Tiempo"
        ax1.scatter(dataFrame['Porcentaje'], dataFrame['Tiempo'],
                    c=kmeans.labels_.astype(float), s=50, alpha=0.5)
        ax1.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

        scatter1 = FigureCanvasTkAgg(figure1, root)
        scatter1.get_tk_widget()
        # .pack(side=tk.LEFT, fill=tk.BOTH)
        scatter1.get_tk_widget().place(x=620, y=180)


def runAB(n):
    cnum = int(n)
    consulta = ""
    url = urlTexField.get()
    canti = cantPetiTexField.get()
    concu = cantConcuTexField.get()

    if url == "":
        url = "uis.edu.co"
        canti = 10
        concu = 10

    if canti == "" and cnum == 1:
        messagebox.showerror(
            message="INGRESE NUMERO DE PETICIONES", title="ERROR!!!")

    if concu == "" and cnum == 2:
        messagebox.showerror(
            message="INGRESE NUMERO DE PETICIONES CONCURRENTES", title="ERROR!!!")
    if n == 0:

        messagebox.showerror(
            message="SELECCIONE EL TIPO DE CONSULTA", title="ERROR!!!")

    if n == 1:
        consulta = "ab -n "+str(canti)+" -e data.csv https://www."+url+"/"
        print(consulta)

    if n == 2:
        consulta = "ab -n "+str(canti)+" -c "+str(concu) + \
            " -e data.csv https://www."+url+"/"
        print(consulta)

    res = os.popen(consulta).read()
    txtRes.delete("0.0", tk.END)
    txtRes.insert(1.0, str(res))
    dataF()


def cleanAll():
    urlTexField.delete(0, 'end')
    cantPetiTexField.delete(0, 'end')
    cantConcuTexField.delete(0, 'end')
    comboTipo.current(0)
    comboClus.current(0)


def runFunt():
    numC = comboTipo.current()
    runAB(numC)

    if numC == 1:
        clustering(dataF())

    if numC == 2:
        regresionLineal(dataF())
    cleanAll()
# estetico


def hide():

    hideforCon()
    hideforSec()


def hideforSec():
    textClus.place_forget()
    comboClus.place_forget()


def hideforCon():
    textConcu.place_forget()
    cantConcuTexField.place_forget()


def comboxTipoFunt(o):
    n = comboTipo.current()

    if n == 1:
        showforSec()

    if n == 2:
        showforCon()


def showforCon():
    textConcu.place(x=535, y=100)
    cantConcuTexField.place(x=670, y=100)
    hideforSec()
    print("forCon")


def showforSec():
    textClus.place(x=535, y=100)
    comboClus.place(x=630, y=100)
    hideforCon()
    print("forSec")


def cAll():
    urlTexField.delete(0, 'end')
    cantPetiTexField.delete(0, 'end')
    cantConcuTexField.delete(0, 'end')
    comboTipo.current(0)
    comboClus.current(0)
    txtRes.delete('1.0', END)
    txtAn.delete('1.0', END)


root = Tk()
root.geometry("1150x1500+100+100")
root.title("APATC-S/O")

main = Frame(root)
main.pack(fill=BOTH, expand=1)

canv = Canvas(main)
canv.pack(side=LEFT, fill=BOTH, expand=1)

scrbar = ttk.Scrollbar(main, orient=VERTICAL, command=canv.yview)
scrbar.pack(side=RIGHT, fill=Y)

canv.configure(yscrollcommand=scrbar.set)
canv.bind('<Configure>', lambda e: canv.configure(
    scrollregion=canv.bbox("all")))

secFrame = Frame(canv)

canv.create_window((0, 0), window=secFrame, anchor="nw")


# button close
btnClose = Button(root, text="QUIT", fg="red", command=root.destroy)
btnClose.place(x=1075, y=0)


# titles
Label(root, text="APLICATIVO PYTHON PARA ANALISIS DE TIEMPOS DE CARGA").place(
    x=200, y=10)
Label(root, text="EN SEGUNDO PLANO SECUENCIALES O CONCURRENTES").place(x=210, y=30)

# url Textfield
Label(root, text="INTRODUSCA LOS DATOS DE LA CONSULTA").place(x=0, y=80)
Label(root, text="URL: ").place(x=0, y=100)

urlTexField = Entry(root, width=20, text="")
urlTexField.place(x=40, y=100)
urlTexField.insert(END, "")

# combboxTipoConsulta
Label(root, text="TIPO: ").place(x=210, y=100)
comboTipo = ttk.Combobox(root,
                         values=[
                             "Seleccionar",
                             "SECUENCIAL",
                             "CONCURRENTE"], width=13)

comboTipo.place(x=250, y=100)
comboTipo.current(0)
comboTipo.bind("<<ComboboxSelected>>", comboxTipoFunt)


# cantida Textfield
Label(root, text="# PETICIONES: ").place(x=375, y=100)
cantPetiTexField = Entry(root, width=5, text="")
cantPetiTexField.place(x=480, y=100)
cantPetiTexField.insert(END, "")


# cantida concurrentes Textfield
textConcu = Label(root, text="# CONCURRENTES: ")
textConcu.place(x=535, y=100)


cantConcuTexField = Entry(root, width=5, text="")
cantConcuTexField.place(x=670, y=100)
cantConcuTexField.insert(END, "")


# Combobox Numero Clusters
textClus = Label(root, text="# CLUSTERS: ")
textClus.place(x=720, y=100)


comboClus = ttk.Combobox(root,
                         values=[
                             "#",
                             "2",
                             "3"], width=10)

comboClus.place(x=815, y=100)
comboClus.current(0)


# button run
btnRun = Button(root, command=runFunt, text="Analizar")
btnRun.place(x=730, y=96)

# textscroll Reporte
Label(root, text="RESULTADOS ").place(x=20, y=150)
txtRes = scrolledtext.ScrolledText(root, width=70, height=10)
txtRes.place(x=20, y=180)

Label(root, text="ANALISIS ").place(x=20, y=360)
txtAn = scrolledtext.ScrolledText(root, width=70, height=10)
txtAn.place(x=20, y=385)

root.after(100, hide)
root.mainloop()
