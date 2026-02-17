# --- Librerías Base ---
import pandas as pd
import numpy as np
import time
import ast

import matplotlib.pyplot as plt
import seaborn as sns
import folium

import osmnx as ox
from shapely.geometry import Polygon

# --- Machine Learning & Estadística ---
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import numpy as np
import matplotlib.pyplot as plt



# 1. Función para convertir el texto/lista en un objeto Polygon
def get_centroid(poly_data):
    try:
        # Si los datos vienen como string, los convertimos a lista
        if isinstance(poly_data, str):
            poly_data = ast.literal_eval(poly_data)
        
        # Creamos el polígono (asumiendo que es la primera lista dentro de la estructura)
        # poly_data[0] accede a la lista de coordenadas
        p = Polygon(poly_data[0])
        
        # Retornamos el centroide (x = longitud, y = latitud)
        return p.centroid.x, p.centroid.y
    except:
        return None, None;



def contar_lugares_cercanos(lat, lon, radio=1000):
    try:
        # Descarga los elementos de interés en el radio definido (metros)
        gdf_pois = ox.features_from_point((lat, lon), tags, dist=radio)
        
        if gdf_pois.empty:
            return 0, []
        
        # Contamos cuántos encontramos y extraemos sus nombres
        cantidad = len(gdf_pois)
        nombres = gdf_pois['name'].dropna().unique().tolist()
        
        return cantidad, nombres
    except Exception:
        # Manejo de casos donde no hay datos en esa zona o error de conexión
        return 0, [];


def analizar_entorno(lat, lon, radio=1000):
    # Inicializamos contadores
    counts = {
        'lugares_cerca_count': 0,
        'Playa_1km': 0,
        'Malls_1km': 0,
        'stadium_1km': 0,
        'park_1km': 0,
        'University_1km': 0
    }
    
    try:
        # Descargamos todos los objetos de interés en el radio de 1km
        features = ox.features_from_point((lat, lon), tags_interes, dist=radio)
        
        if features.empty:
            return pd.Series(counts)

        # Total de lugares encontrados
        counts['lugares_cerca_count'] = len(features)
        
        # Clasificamos según las columnas de OSM
        if 'natural' in features.columns:
            counts['Playa_1km'] = features['natural'].eq('beach').sum()
        
        if 'shop' in features.columns:
            counts['Malls_1km'] = features['shop'].eq('mall').sum()
            
        if 'leisure' in features.columns:
            counts['park_1km'] = features['leisure'].eq('park').sum()
            counts['stadium_1km'] += features['leisure'].eq('stadium').sum()
            
        if 'amenity' in features.columns:
            counts['University_1km'] = features['amenity'].eq('university').sum()
            counts['stadium_1km'] += features['amenity'].eq('stadium').sum()

    except Exception as e:
        # En caso de que no haya resultados, OSMnx a veces lanza error
        pass
        
    return pd.Series(counts);


def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia en kilómetros entre dos puntos geográficos.
    """
    R = 6371.0 # Radio aproximado de la Tierra en km
    
    # Convertir coordenadas a radianes
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    # Fórmula de Haversine
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c;


# Función para listar pares altamente correlacionados
def get_high_correlations(df, threshold=0.9):
    corr = df.corr().abs()
    # Seleccionar solo la parte superior de la matriz
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    
    # Encontrar índices con correlación mayor al umbral
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Crear un resumen de los pares
    pairs = []
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold]
        for index, value in high_corr.items():
            pairs.append((index, col, round(value, 4)))
            
    return pd.DataFrame(pairs, columns=['Variable 1', 'Variable 2', 'Correlación']), to_drop;




def plot_cluster_radar(df, features):
    # 1. Calculamos el promedio por clúster para las variables de interés
    df_grouped = df.groupby('cluster')[features].mean()
    
    # 2. Normalización Min-Max para que todas las variables luzcan bien en el radar (rango 0-1)
    df_min_max = (df_grouped - df_grouped.min()) / (df_grouped.max() - df_grouped.min())
    
    # 3. Configuración del gráfico de Radar
    categories = list(df_min_max.columns)
    N = len(categories)
    
    # Ángulos para cada eje
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # Cerrar el círculo
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Colores para los clústeres
    colors = plt.cm.get_cmap("viridis", len(df_min_max))
    
    for i, (idx, row) in enumerate(df_min_max.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1] # Cerrar el círculo
        
        # Dibujar la línea y el área
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Clúster {idx}', color=colors(i))
        ax.fill(angles, values, color=colors(i), alpha=0.25)
    
    # Ajustes estéticos
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
    plt.ylim(0, 1)
    
    plt.title("Perfil de Características por Clúster (Radar)", size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show();


def plot_pca_combined_biplot(df, pca, features, id_col='ID_F'):
    plt.figure(figsize=(14, 9))
    
    # 1. Graficar los puntos (parqueaderos) coloreados por clúster
    sns.scatterplot(
        x='pca_1', y='pca_2', hue='cluster', style='cluster',
        data=df, palette='viridis', s=150, alpha=0.7, edgecolor='w'
    )
    
    # 2. Factor de escala para las flechas (para que coincidan con la escala de los puntos)
    # Ajustamos las flechas para que cubran el 80% del rango de los puntos
    scale_x = df['pca_1'].abs().max() * 0.8
    scale_y = df['pca_2'].abs().max() * 0.8
    
    coef = np.transpose(pca.components_[0:2, :])
    
    # 3. Dibujar las flechas de las variables
    for i in range(coef.shape[0]):
        plt.arrow(0, 0, coef[i,0]*scale_x, coef[i,1]*scale_y, 
                  color='red', alpha=0.6, head_width=scale_x*0.02, head_length=scale_x*0.03)
        plt.text(coef[i,0]*scale_x*1.1, coef[i,1]*scale_y*1.1, features[i], 
                 color='darkred', ha='center', va='center', fontweight='bold', fontsize=10)

    # 4. Anotar los nombres de los parqueaderos (opcional, para identificar outliers)
    for i in range(df.shape[0]):
        plt.text(df.pca_1[i]+0.05, df.pca_2[i]+0.05, df[id_col][i], 
                 fontsize=8, alpha=0.6)

    # Formato final
    plt.axhline(0, color='black', lw=1, linestyle='--')
    plt.axvline(0, color='black', lw=1, linestyle='--')
    plt.title("Biplot Integrado: Parqueaderos y su relación con las Variables", size=15)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%) - 'Dimensión de Volumen/Negocio'")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%) - 'Dimensión de Entorno'")
    plt.grid(True, alpha=0.2)
    plt.legend(title="Clúster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show();


def expandir_presencia_carros(row):
    # Generamos un rango desde la hora de inicio hasta la hora de fin
    # Usamos 'h' para evitar el error de frecuencia en versiones nuevas
    horas_rango = pd.date_range(
        start=row['start'].floor('h'), 
        end=row['end'].floor('h'), 
        freq='h'
    )
    
    intervalos = []
    for h in horas_rango:
        intervalos.append({
            'Fecha': h.date(),
            'Hora': h.hour,
            'JOIN_ID': row['JOIN_ID']
        })
    return intervalos;