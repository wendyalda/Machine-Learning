import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

NOMBRES_CLASES = [
    "Perro", "Caballo", "Elefante", "Mariposa", "Gallina", 
    "Gato", "Vaca", "Oveja", "Araña", "Ardilla"
]

def graficar_probabilidades(probabilidades_dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    clases = list(probabilidades_dict.keys())
    valores = list(probabilidades_dict.values())
    
    colores = ['#2ecc71' if v == max(valores) else '#3498db' for v in valores]
    bars = ax.barh(clases, valores, color=colores)
    ax.set_xlabel('Probabilidad (%)', fontsize=11, fontweight='bold')
    ax.set_title('Distribución de Probabilidades (Softmax)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        if width > 1.0:
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                    va='center', ha='left', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    return fig

def graficar_curvas_entrenamiento(df_historial):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Curva de Exactitud (Accuracy)
    ax1.plot(df_historial['accuracy'], label='Entrenamiento', color='#2980b9', linewidth=2)
    if 'val_accuracy' in df_historial.columns:
        ax1.plot(df_historial['val_accuracy'], label='Validación', color='#e67e22', linewidth=2)
    ax1.set_title('Evolución de Exactitud (Accuracy)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Exactitud')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Curva de Pérdida (Loss)
    ax2.plot(df_historial['loss'], label='Entrenamiento', color='#2980b9', linewidth=2)
    if 'val_loss' in df_historial.columns:
        ax2.plot(df_historial['val_loss'], label='Validación', color='#e67e22', linewidth=2)
    ax2.set_title('Evolución de Pérdida (Loss)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig