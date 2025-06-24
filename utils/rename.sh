#!/bin/bash

# Script para renombrar archivos reemplazando espacios por guiones bajos
# Uso: ./rename_spaces.sh [directorio]

# Función para mostrar ayuda
show_help() {
    echo "Uso: $0 [directorio]"
    echo ""
    echo "Este script renombra todos los archivos en el directorio especificado"
    echo "y sus subdirectorios, reemplazando los espacios ' ' por guiones bajos '_'"
    echo ""
    echo "Parámetros:"
    echo "  directorio    Directorio a procesar (opcional, por defecto: directorio actual)"
    echo ""
    echo "Opciones:"
    echo "  -h, --help    Mostrar esta ayuda"
    echo ""
    echo "Ejemplos:"
    echo "  $0                    # Procesar directorio actual"
    echo "  $0 /home/user/docs    # Procesar directorio específico"
}

# Verificar si se solicita ayuda
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Establecer directorio objetivo
TARGET_DIR="${1:-.}"

# Verificar que el directorio existe
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: El directorio '$TARGET_DIR' no existe."
    exit 1
fi

# Convertir a ruta absoluta
TARGET_DIR=$(readlink -f "$TARGET_DIR")

echo "Procesando directorio: $TARGET_DIR"
echo "Buscando archivos con espacios en el nombre..."

# Contador de archivos renombrados
renamed_count=0

# Función para renombrar archivos
rename_files() {
    local current_dir="$1"
    
    # Procesar archivos en el directorio actual
    find "$current_dir" -maxdepth 1 -type f -name "* *" -print0 | while IFS= read -r -d '' file; do
        # Obtener el directorio padre y el nombre del archivo
        dir_path=$(dirname "$file")
        old_name=$(basename "$file")
        new_name="${old_name// /_}"
        
        # Nueva ruta completa
        new_path="$dir_path/$new_name"
        
        # Verificar si el archivo de destino ya existe
        if [ -e "$new_path" ]; then
            echo "Advertencia: '$new_path' ya existe. Omitiendo '$file'"
        else
            # Renombrar el archivo
            if mv "$file" "$new_path" 2>/dev/null; then
                echo "Renombrado: '$old_name' -> '$new_name'"
                ((renamed_count++))
            else
                echo "Error: No se pudo renombrar '$file'"
            fi
        fi
    done
}

# Procesar todos los directorios recursivamente
# Usamos un enfoque que procesa primero los archivos más profundos
find "$TARGET_DIR" -type d -print0 | sort -rz | while IFS= read -r -d '' dir; do
    rename_files "$dir"
done

# También renombrar directorios con espacios (de más profundo a menos profundo)
echo ""
echo "Renombrando directorios con espacios..."

find "$TARGET_DIR" -type d -name "* *" -print0 | sort -rz | while IFS= read -r -d '' dir; do
    # Obtener el directorio padre y el nombre del directorio
    parent_dir=$(dirname "$dir")
    old_dir_name=$(basename "$dir")
    new_dir_name="${old_dir_name// /_}"
    
    # Nueva ruta completa
    new_dir_path="$parent_dir/$new_dir_name"
    
    # Verificar si el directorio de destino ya existe
    if [ -e "$new_dir_path" ]; then
        echo "Advertencia: '$new_dir_path' ya existe. Omitiendo '$dir'"
    else
        # Renombrar el directorio
        if mv "$dir" "$new_dir_path" 2>/dev/null; then
            echo "Directorio renombrado: '$old_dir_name' -> '$new_dir_name'"
        else
            echo "Error: No se pudo renombrar el directorio '$dir'"
        fi
    fi
done

echo ""
echo "Proceso completado."
echo "Archivos procesados en: $TARGET_DIR"

# Mostrar resumen
total_files_with_spaces=$(find "$TARGET_DIR" -type f -name "* *" 2>/dev/null | wc -l)
total_dirs_with_spaces=$(find "$TARGET_DIR" -type d -name "* *" 2>/dev/null | wc -l)

if [ "$total_files_with_spaces" -eq 0 ] && [ "$total_dirs_with_spaces" -eq 0 ]; then
    echo "✓ Todos los archivos y directorios han sido procesados correctamente."
else
    echo "⚠ Aún quedan $total_files_with_spaces archivos y $total_dirs_with_spaces directorios con espacios."
    echo "Esto puede deberse a conflictos de nombres o errores de permisos."
fi

