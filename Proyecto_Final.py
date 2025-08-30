import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import requests
import json
from datetime import datetime
import hashlib

# Configuración de la página
st.set_page_config(
    page_title="Análisis FIFA Qatar 2022",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funciones de autenticación
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    # Credenciales simples como solicita la tarea
    return username == "admin" and password == "admin"

# Función para cargar datos con caché
@st.cache_data
def load_qatar_data():
    """Carga los datos de Qatar 2022 desde el archivo CSV"""
    try:
        # En un caso real, cargarías desde el archivo
        # df = pd.read_csv('player_stats_qatar2022.csv')
        # Para esta demostración, crearemos datos simulados basados en el CSV
        np.random.seed(42)
        
        # Crear datos simulados basados en las columnas del CSV real
        n_players = 100
        
        data = {
            'player_id': range(1, n_players + 1),
            'team_id': np.random.randint(1, 33, n_players),
            'match_id': np.random.randint(1, 65, n_players),
            'goal': np.random.poisson(0.3, n_players),
            'assist': np.random.poisson(0.2, n_players),
            'pass': np.random.normal(45, 15, n_players).astype(int),
            'pass_succeeded': np.random.normal(38, 12, n_players).astype(int),
            'shot_on_target': np.random.poisson(1.5, n_players),
            'shot_off_target': np.random.poisson(1.2, n_players),
            'total_shot': np.random.poisson(2.8, n_players),
            'tackle': np.random.poisson(2.1, n_players),
            'tackle_succeeded': np.random.poisson(1.4, n_players),
            'aerial_duel': np.random.poisson(3.2, n_players),
            'aerial_duel_succeeded': np.random.poisson(1.8, n_players),
            'foul': np.random.poisson(1.1, n_players),
            'yellow_card': np.random.poisson(0.15, n_players),
            'red_card': np.random.poisson(0.02, n_players),
            'play_time': np.random.normal(75, 20, n_players).astype(int),
            'rating': np.random.normal(6.8, 0.8, n_players).round(1)
        }
        
        # Ajustar valores negativos
        for col in data:
            if col != 'player_id':
                data[col] = np.maximum(data[col], 0)
        
        # Calcular pass_accuracy
        data['pass_accuracy'] = (data['pass_succeeded'] / np.maximum(data['pass'], 1) * 100).round(1)
        
        df = pd.DataFrame(data)
        
        # Crear categorías de rendimiento para clasificación
        df['performance_category'] = pd.cut(df['rating'], 
                                          bins=[0, 6.0, 7.0, 8.0, 10.0], 
                                          labels=['Bajo', 'Medio', 'Alto', 'Excelente'])
        
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

# Función para obtener datos de API externa (simulada)
@st.cache_data
def get_fifa_rankings():
    """Simula datos de ranking FIFA"""
    teams = ['Argentina', 'Francia', 'Brasil', 'Inglaterra', 'Países Bajos', 
             'Croacia', 'Italia', 'Portugal', 'España', 'Marruecos']
    
    rankings = {
        'team': teams,
        'ranking': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'points': [1840, 1820, 1810, 1800, 1790, 1780, 1770, 1760, 1750, 1740],
        'confederation': ['CONMEBOL', 'UEFA', 'CONMEBOL', 'UEFA', 'UEFA', 
                         'UEFA', 'UEFA', 'UEFA', 'UEFA', 'CAF']
    }
    
    return pd.DataFrame(rankings)

# Función para crear modelo de clasificación
@st.cache_data
def create_classification_model(df):
    """Crea y entrena un modelo de clasificación para predecir rendimiento"""
    try:
        # Preparar características
        features = ['goal', 'assist', 'pass_accuracy', 'shot_on_target', 
                   'tackle_succeeded', 'aerial_duel_succeeded', 'play_time']
        
        X = df[features].fillna(0)
        y = df['performance_category'].fillna('Medio')
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Métricas
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        return model, scaler, train_score, test_score, features
    
    except Exception as e:
        st.error(f"Error creando modelo: {e}")
        return None, None, 0, 0, []

# Función de login
def login_page():
    st.title("🔐 Inicio de Sesión")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Acceso al Sistema de Análisis FIFA Qatar 2022")
        
        username = st.text_input("👤 Usuario", placeholder="Ingrese su usuario")
        password = st.text_input("🔒 Contraseña", type="password", placeholder="Ingrese su contraseña")
        
        st.info("💡 **Credenciales de prueba:**\n- Usuario: admin\n- Contraseña: admin")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn2:
            if st.button("🚀 Iniciar Sesión", use_container_width=True):
                if authenticate(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("✅ ¡Inicio de sesión exitoso!")
                    st.rerun()
                else:
                    st.error("❌ Credenciales incorrectas")

# Página principal con métricas
def dashboard_page():
    st.title("📊 Dashboard - FIFA Qatar 2022")
    
    # Cargar datos
    df_qatar = load_qatar_data()
    df_rankings = get_fifa_rankings()
    
    if df_qatar.empty:
        st.error("No se pudieron cargar los datos")
        return
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Jugadores", len(df_qatar))
    
    with col2:
        avg_rating = df_qatar['rating'].mean()
        st.metric("⭐ Rating Promedio", f"{avg_rating:.1f}")
    
    with col3:
        total_goals = df_qatar['goal'].sum()
        st.metric("⚽ Total Goles", total_goals)
    
    with col4:
        avg_passes = df_qatar['pass'].mean()
        st.metric("📤 Pases Promedio", f"{avg_passes:.0f}")
    
    st.markdown("---")
    
    # Gráficos principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Distribución de Ratings")
        fig_hist = px.histogram(df_qatar, x='rating', nbins=20, 
                               title="Distribución de Ratings de Jugadores")
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Goles vs Asistencias")
        fig_scatter = px.scatter(df_qatar, x='goal', y='assist', 
                                color='rating', size='play_time',
                                title="Relación Goles vs Asistencias")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Tabla de rankings FIFA
    st.subheader("🏆 Rankings FIFA (Top 10)")
    st.dataframe(df_rankings, use_container_width=True)
    
    # Gráfico de rankings
    fig_rankings = px.bar(df_rankings, x='team', y='points',
                         color='confederation',
                         title="Puntos FIFA por Equipo")
    fig_rankings.update_xaxes(tickangle=45)
    st.plotly_chart(fig_rankings, use_container_width=True)

# Página de análisis avanzado y modelo ML
def analysis_page():
    st.title("🔬 Análisis Avanzado y Modelo de Clasificación")
    
    df_qatar = load_qatar_data()
    
    if df_qatar.empty:
        st.error("No se pudieron cargar los datos")
        return
    
    # Crear modelo
    model, scaler, train_score, test_score, features = create_classification_model(df_qatar)
    
    if model is None:
        st.error("No se pudo crear el modelo de clasificación")
        return
    
    # Información del modelo
    st.subheader("🤖 Modelo de Clasificación de Rendimiento")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Precisión Entrenamiento", f"{train_score:.3f}")
    
    with col2:
        st.metric("🎯 Precisión Prueba", f"{test_score:.3f}")
    
    with col3:
        st.metric("🔢 Características", len(features))
    
    # Importancia de características
    st.subheader("📋 Importancia de Características")
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig_importance = px.bar(feature_importance, x='importance', y='feature',
                           orientation='h', title="Importancia de Características")
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Predictor interactivo
    st.subheader("🎮 Predictor de Rendimiento")
    st.write("Ajusta los valores para predecir la categoría de rendimiento:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        goal_pred = st.slider("Goles", 0, 10, 2)
        assist_pred = st.slider("Asistencias", 0, 8, 1)
        pass_accuracy_pred = st.slider("Precisión de Pases (%)", 0.0, 100.0, 80.0)
        shot_target_pred = st.slider("Tiros a Portería", 0, 10, 3)
    
    with col2:
        tackle_succ_pred = st.slider("Entradas Exitosas", 0, 10, 2)
        aerial_succ_pred = st.slider("Duelos Aéreos Exitosos", 0, 10, 2)
        play_time_pred = st.slider("Tiempo de Juego", 0, 90, 75)
    
    # Realizar predicción
    if st.button("🔮 Predecir Rendimiento"):
        input_data = np.array([[goal_pred, assist_pred, pass_accuracy_pred,
                               shot_target_pred, tackle_succ_pred, 
                               aerial_succ_pred, play_time_pred]])
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        st.success(f"🎯 **Predicción:** {prediction}")
        
        # Mostrar probabilidades
        prob_df = pd.DataFrame({
            'Categoría': model.classes_,
            'Probabilidad': probability
        }).sort_values('Probabilidad', ascending=False)
        
        fig_prob = px.bar(prob_df, x='Categoría', y='Probabilidad',
                         title="Probabilidades por Categoría")
        st.plotly_chart(fig_prob, use_container_width=True)
    
    # Análisis de correlaciones
    st.subheader("🔗 Matriz de Correlaciones")
    
    numeric_cols = ['goal', 'assist', 'pass_accuracy', 'shot_on_target', 
                    'total_shot', 'tackle', 'rating']
    correlation_matrix = df_qatar[numeric_cols].corr()
    
    fig_corr = px.imshow(correlation_matrix, 
                        title="Matriz de Correlaciones",
                        color_continuous_scale="RdBu")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Tabla de datos filtrable
    st.subheader("📋 Datos Detallados")
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_rating = st.slider("Rating Mínimo", 
                              float(df_qatar['rating'].min()), 
                              float(df_qatar['rating'].max()), 
                              float(df_qatar['rating'].min()))
    
    with col2:
        min_goals = st.slider("Goles Mínimos", 
                             int(df_qatar['goal'].min()), 
                             int(df_qatar['goal'].max()), 
                             int(df_qatar['goal'].min()))
    
    with col3:
        performance_filter = st.selectbox("Categoría de Rendimiento", 
                                        ['Todos'] + list(df_qatar['performance_category'].unique()))
    
    # Aplicar filtros
    filtered_df = df_qatar.copy()
    filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
    filtered_df = filtered_df[filtered_df['goal'] >= min_goals]
    
    if performance_filter != 'Todos':
        filtered_df = filtered_df[filtered_df['performance_category'] == performance_filter]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Botones de exportación (simulados)
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🖨️ Imprimir Página"):
            st.info("📄 Funcionalidad de impresión activada (simulada)")
    
    with col2:
        if st.button("📄 Exportar a PDF"):
            st.info("💾 Exportación a PDF iniciada (simulada)")

# Función principal
def main():
    # Inicializar estado de sesión
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Sidebar para navegación
    with st.sidebar:
        st.title("⚽ FIFA Qatar 2022")
        st.write(f"👋 Bienvenido, {st.session_state.get('username', 'Usuario')}")
        
        # Menú de navegación
        page = st.selectbox(
            "📋 Navegar",
            ["📊 Dashboard", "🔬 Análisis Avanzado"],
            index=0
        )
        
        st.markdown("---")
        
        # Información de la sesión
        st.subheader("ℹ️ Información")
        st.write(f"🕐 Sesión iniciada: {datetime.now().strftime('%H:%M:%S')}")
        
        # Botón de cerrar sesión
        if st.button("🚪 Cerrar Sesión", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
    
    # Mostrar página seleccionada
    if page == "📊 Dashboard":
        dashboard_page()
    elif page == "🔬 Análisis Avanzado":
        analysis_page()

if __name__ == "__main__":
    main()