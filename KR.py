import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mplcursors
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import Label, Entry, Button, messagebox

# Функция для запуска анализа
def run_analysis():
    # Открытие диалогового окна для выбора файла Excel
    path_to_file = askopenfilename()

    # Чтение данных из Excel таблицы
    df = pd.read_excel(path_to_file)

    # Выделяем только столбец с марками машин
    marks = df.iloc[:, 0]

    # Удаление столбца 'Марка'
    df = df.drop(columns=['Марка/Конструкт'])

    # Выделяем только столбцы с оценками
    X = df.iloc[:, 1:]

    # Вычисление корреляции для всех характеристик
    correlation_matrix = X.corr()

    # Проверка наличия файла
    if os.path.isfile("C:/Users/olegf/Desktop/correlation_matrix.xlsx"):
        # Если файл существует, удаляем его
        os.remove("C:/Users/olegf/Desktop/correlation_matrix.xlsx")

    correlation_matrix.to_excel("C:/Users/olegf/Desktop/correlation_matrix.xlsx", index=True)

    # Нормализуем данные
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Определяем количество компонент
    n_components = int(entry_components.get())

    # Создаем объект PCA
    pca = PCA(n_components=n_components)

    # Применяем PCA к данным
    X_pca = pca.fit_transform(X)

    # Выводим доли объясненной дисперсии каждой компоненты
    print("Объясненный коэффициент дисперсии:", pca.explained_variance_ratio_)

    # Выводим значения компонент для каждой строки
    print("Значения PCA для каждой строки:")
    for i, (row_name, row) in enumerate(zip(marks, X_pca)):
        print(f"Строка {i + 1} ({row_name}): {row}")

    # Извлекаем значения первых двух компонент
    PC1 = X_pca[:, 0]
    PC2 = X_pca[:, 1]

    # Создаем массив случайных цветов для каждой точки
    random_colors = np.random.rand(len(PC1), 3)

    # Создаем scatter plot для PCA
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.scatter(PC1, PC2, c=random_colors)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Scatter Plot')

    # Добавляем подписи для каждой точки (используем столбец с марками машин)
    for i, txt in enumerate(marks):
        plt.text(PC1[i], PC2[i], txt, fontsize=8, ha='center', va='bottom')

    # Функция для отображения значения точки в подсказке в PCA графике
    def annotation_func_pca(sel):
        index = sel.index
        x = PC1[index]
        y = PC2[index]
        value = X_pca[index]
        sel.annotation.set_text(f"Value: {value}")
        sel.annotation.set_position((x, y))

    # Создаем подсказки при наведении на точки в PCA графике
    cursors_pca = mplcursors.cursor(hover=True)
    cursors_pca.connect("add", annotation_func_pca)

    # Кластерный анализ с использованием K-means
    num_clusters = int(entry_clusters.get())
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    cluster_centers = kmeans.cluster_centers_

    # Создаем scatter plot для кластерного анализа
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(PC1, PC2, c=cluster_labels, cmap='viridis', alpha=0.5)
    centers = plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                          marker='x', color='red', label='Cluster Centers')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('K-means Clustering')

    # Добавляем подписи для каждой точки (используем столбец с марками машин)
    for i, txt in enumerate(marks):
        plt.text(PC1[i], PC2[i], txt, fontsize=8, ha='center', va='bottom')

    # Функция для отображения номера кластера в подсказке в графике кластерного анализа
    def annotation_func_cluster(sel):
        if sel.artist == centers:
            cluster = sel.index
            x = cluster_centers[cluster, 0]
            y = cluster_centers[cluster, 1]
            sel.annotation.set_text(f"Cluster Center: {cluster_centers[cluster]}\nCluster: {cluster}")
            sel.annotation.set_position((x, y))
        elif sel.artist == scatter:
            index = sel.index
            x = PC1[index]
            y = PC2[index]
            cluster = cluster_labels[index]
            sel.annotation.set_text(f"Cluster: {cluster}")
            sel.annotation.set_position((x, y))
        else:
            sel.annotation.set_visible(False)

    # Расчет анализа согласованности (ассиметрии) по Р. Саммерсу
    def calculate_symmetry():
        # Выбор первых двух компонент
        PC1 = X_pca[:, 0]
        PC2 = X_pca[:, 1]

        # Вычисление анализа согласованности по Р. Саммерсу
        asymmetry = np.abs(np.mean(PC2 ** 3) / np.mean(PC1 ** 2) ** (3 / 2))

        # Вывод результата
        messagebox.showinfo("Symmetry Analysis", f"Symmetry (R. Summers): {asymmetry}")

    # Добавление кнопки для запуска анализа согласованности по Р. Саммерсу
    btn_symmetry = Button(window, text="Calculate Symmetry", command=calculate_symmetry)
    btn_symmetry.pack()

    # Создаем подсказки при наведении на точки в графике кластерного анализа
    cursors_cluster = mplcursors.cursor(hover=True)
    cursors_cluster.connect("add", annotation_func_cluster)

    # Создаем график bar plot для долей объясненной дисперсии PCA
    plt.subplot(1, 3, 3)
    explained_variance_ratio = pca.explained_variance_ratio_
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio for PCA')

    # Отображаем графики
    plt.tight_layout()
    plt.show()

    # Показываем окно с сообщением о завершении анализа
    messagebox.showinfo("Analysis Complete", "Repertory Grid Analysis Finished.")

    # Закрытие окна после завершения анализа
    window.destroy()

# Создание окна
window = Tk()
window.title("Cluster Analysis")

# Надпись и поле ввода для количества компонент
label_components = Label(window, text="Number of Components:")
label_components.pack()
entry_components = Entry(window)
entry_components.pack()

# Надпись и поле ввода для количества кластеров
label_clusters = Label(window, text="Number of Clusters:")
label_clusters.pack()
entry_clusters = Entry(window)
entry_clusters.pack()

# Кнопка запуска анализа
btn_run = Button(window, text="Run Analysis", command=run_analysis)
btn_run.pack()

# Кнопка завершения анализа
btn_finish = Button(window, text="Finish Analysis", command=window.quit)
btn_finish.pack()

# Запуск основного цикла окна
window.mainloop()