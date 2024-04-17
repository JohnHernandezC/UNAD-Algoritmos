# Paso 1: Análisis Exploratorio de Datos
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv('heart_cleveland_upload.csv')

# Verificar las primeras filas del dataframe
print(data.head())

# Explorar la distribución de la variable objetivo
sns.countplot(x='condition', data=data)
plt.title('Distribución de la condición de enfermedad cardíaca')
plt.show()

# Explorar la correlación entre variables
sns.heatmap(data.corr(), annot=True)
plt.title('Correlación entre variables')
plt.show()

# Paso 2: Preprocesamiento de Datos
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Tratar valores faltantes
imputer = SimpleImputer(strategy='mean')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Dividir variables predictoras y variable objetivo
X = data_filled.drop('condition', axis=1)
y = data_filled['condition']

# Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 3: Selección de Características
from sklearn.feature_selection import SelectKBest, f_classif

# Seleccionar las 5 características más relevantes
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_scaled, y)

# Paso 4: Dividir el dataset en Train y Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Paso 5: Entrenar el Modelo
from sklearn.linear_model import LogisticRegression

# Configurar los hiperparámetros
model = LogisticRegression(max_iter=1000)

# Entrenar el modelo
model.fit(X_train, y_train)

# Paso 6: Evaluar el Desempeño del Modelo
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Paso 7: Gráficas para Visualizar Resultados
from sklearn.metrics import plot_confusion_matrix

# Matriz de Confusión
plot_confusion_matrix(model, X_test, y_test)
plt.title('Matriz de Confusión')
plt.show()

# Paso 8: Interpretación y Documentación de Resultados
# Se pueden analizar las métricas obtenidas, la matriz de confusión y la importancia de las características seleccionadas para interpretar el desempeño del modelo.
