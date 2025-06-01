import streamlit as st
import os
from comparaciones import show_beat_spectrum
import matplotlib.pyplot as plt
import librosa
import tempfile
import io

def get_file_extension(file):
    """Detecta la extensión del archivo"""
    if file is not None:
        return os.path.splitext(file.name)[1].lower()
    return None


def load_audios_from_streamlit_files(file1, file2):
    """
    Carga archivos de audio desde objetos de archivo de Streamlit usando librosa
    
    Opción 1: Usando archivos temporales (recomendado para mayor compatibilidad)
    """
    try:
        # Crear archivos temporales
        with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(file1)) as temp_file1:
            temp_file1.write(file1.getvalue())
            temp_path1 = temp_file1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(file2)) as temp_file2:
            temp_file2.write(file2.getvalue())
            temp_path2 = temp_file2.name
        
        fig1, fig2, fig3 = show_beat_spectrum(temp_path1, temp_path2)
        
        st.subheader("📊 Análisis de Beat Spectrums")
        st.pyplot(fig1)
        plt.close(fig1)  # Liberar memoria
        
        if fig2:
            st.subheader("🎵 Errores Rítmicos")
            st.pyplot(fig2)
            plt.close(fig2)  # Liberar memoria
        
        if fig3:
            st.subheader("🔍 Análisis Detallado de Onsets")
            st.pyplot(fig3)
            plt.close(fig3)  # Liberar memoria
        
        # Limpiar archivos temporales
        os.unlink(temp_path1)
        os.unlink(temp_path2)
        
    except Exception as e:
        st.error(f"Error al cargar los archivos de audio: {str(e)}")
        return None, None, None

def load_audios_from_bytes(file1, file2):
    """
    Opción 2: Cargar directamente desde bytes (funciona con algunos formatos)
    """
    try:
        # Obtener los bytes de los archivos
        file1_bytes = io.BytesIO(file1.getvalue())
        file2_bytes = io.BytesIO(file2.getvalue())
        
        # Cargar con librosa directamente desde bytes
        reference_audio, sampling_rate = librosa.load(file1_bytes)
        live_audio, _ = librosa.load(file2_bytes, sr=sampling_rate)
        
        return reference_audio, live_audio, sampling_rate
        
    except Exception as e:
        st.error(f"Error al cargar desde bytes: {str(e)}")
        # Fallback a archivos temporales
        return load_audios_from_streamlit_files(file1, file2)
    """Detecta la extensión del archivo"""
    if file is not None:
        return os.path.splitext(file.name)[1].lower()
    return None

def main():
    st.title("🎵 Subida de Archivos de Audio")
    st.write("Sube dos archivos de audio (.mp3, .wav o .mid) para procesar")
    
    # Configurar los tipos de archivo permitidos
    allowed_types = ["mp3", "wav", "mid"]
    
    # Crear dos file uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Archivo 1")
        file1 = st.file_uploader(
            "Selecciona el primer archivo",
            type=allowed_types,
            key="file1"
        )
        
        # Mostrar información del primer archivo
        if file1 is not None:
            ext1 = get_file_extension(file1)
            st.success(f"✅ Archivo cargado: {file1.name}")
            st.info(f"📁 Extensión detectada: {ext1}")
            st.write(f"📊 Tamaño: {file1.size} bytes")
    
    with col2:
        st.subheader("Archivo 2")
        file2 = st.file_uploader(
            "Selecciona el segundo archivo",
            type=allowed_types,
            key="file2"
        )
        
        # Mostrar información del segundo archivo
        if file2 is not None:
            ext2 = get_file_extension(file2)
            st.success(f"✅ Archivo cargado: {file2.name}")
            st.info(f"📁 Extensión detectada: {ext2}")
            st.write(f"📊 Tamaño: {file2.size} bytes")
    
    # Separador visual
    st.divider()
    
    # Botón de acción (solo habilitado si hay dos archivos)
    if file1 is not None and file2 is not None:
        st.success("🎉 ¡Ambos archivos están listos!")
        
        # Mostrar resumen
        col1_summary, col2_summary = st.columns(2)
        with col1_summary:
            st.write("**Archivo 1:**")
            st.write(f"- Nombre: {file1.name}")
            st.write(f"- Extensión: {get_file_extension(file1)}")
        
        with col2_summary:
            st.write("**Archivo 2:**")
            st.write(f"- Nombre: {file2.name}")
            st.write(f"- Extensión: {get_file_extension(file2)}")
        
        # Botón principal de acción
        if st.button("🚀 Procesar Archivos", type="primary", use_container_width=True):
            # Procesar archivos con librosa
            with st.spinner("Cargando y procesando archivos de audio..."):
                
                # Cargar los audios usando librosa
                reference_audio, live_audio, sampling_rate = load_audios_from_streamlit_files(file1, file2)
                
                if reference_audio is not None and live_audio is not None:
                    st.success("✅ ¡Archivos de audio cargados exitosamente!")
                    st.balloons()
                    
                    # Mostrar información de los audios cargados
                    st.write("**Información de los audios:**")
                    
                    col1_info, col2_info = st.columns(2)
                    with col1_info:
                        st.write(f"**{file1.name}:**")
                        st.write(f"- Duración: {len(reference_audio)/sampling_rate:.2f} segundos")
                        st.write(f"- Muestras: {len(reference_audio):,}")
                        st.write(f"- Sample Rate: {sampling_rate} Hz")
                    
                    with col2_info:
                        st.write(f"**{file2.name}:**")
                        st.write(f"- Duración: {len(live_audio)/sampling_rate:.2f} segundos")
                        st.write(f"- Muestras: {len(live_audio):,}")
                        st.write(f"- Sample Rate: {sampling_rate} Hz")
                    
                    st.write("---")
                    st.write("**Procesamiento completado:** Los archivos están listos para análisis adicional")
                    
                        
                    # Aquí puedes agregar más análisis con librosa:
                
                else:
                    st.error("❌ Error al procesar los archivos de audio")
    
    else:
        # Estado cuando faltan archivos
        missing_files = []
        if file1 is None:
            missing_files.append("Archivo 1")
        if file2 is None:
            missing_files.append("Archivo 2")
        
        st.warning(f"⚠️ Faltan archivos: {', '.join(missing_files)}")
        st.button("🚀 Procesar Archivos", disabled=True, use_container_width=True)
    
    # Información adicional
    with st.expander("Información adicional"):
        st.write("**Formatos soportados:**")
        st.write("- **MP3**: Formato de audio comprimido")
        st.write("- **WAV**: Formato de audio sin compresión")
        st.write("- **MID**: Archivos MIDI")
        st.write("\n**Procesamiento:**")
        st.write("- Los archivos se cargan usando librosa para análisis de audio")
        st.write("- Se crean archivos temporales seguros para el procesamiento")
        st.write("- Los archivos temporales se eliminan automáticamente")
        st.write("\n**Instrucciones:**")
        st.write("1. Sube dos archivos de audio en los formatos permitidos")
        st.write("2. La aplicación detectará automáticamente las extensiones")
        st.write("3. El botón de procesamiento se habilitará cuando ambos archivos estén cargados")
        st.write("4. Los archivos se procesarán con librosa para análisis de audio")

if __name__ == "__main__":
    main()