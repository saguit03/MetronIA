#!/usr/bin/env python3
"""
Script de ejemplo para validar el analizador de mutaciones usando matrices de confusión.

Este script puede ejecutarse independientemente para analizar la efectividad del 
analizador comparando los cambios esperados con los detectados.
"""

import sys
import argparse
from pathlib import Path
from analyzers.validation_analyzer import MutationValidationAnalyzer


def main():
    """Función principal del script de validación."""
    parser = argparse.ArgumentParser(
        description="Validación del analizador de mutaciones usando matrices de confusión",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Validar todas las mutaciones de un MIDI específico
  python validate_analyzer.py --midi Tarrega_Gran_Vals

  # Validar una mutación específica
  python validate_analyzer.py --midi Tarrega_Gran_Vals --category timing_errors --mutation note_too_late

  # Especificar directorio de resultados personalizado
  python validate_analyzer.py --midi Acordai-100 --results-dir custom_results/

  # Solo generar reporte sin gráficos
  python validate_analyzer.py --midi Tarrega_Gran_Vals --no-plots
        """
    )
    
    parser.add_argument(
        '--midi', 
        required=True,
        help='Nombre del MIDI de referencia (sin extensión)'
    )
    
    parser.add_argument(
        '--category',
        help='Categoría específica de mutación a validar (opcional)'
    )
    
    parser.add_argument(
        '--mutation',
        help='Nombre específico de mutación a validar (requiere --category)'
    )
    
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Directorio de resultados (por defecto: results)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='No generar gráficos, solo reportes de texto'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Directorio específico para guardar resultados de validación (opcional)'
    )
    
    args = parser.parse_args()
    
    # Validaciones de argumentos
    if args.mutation and not args.category:
        print("❌ Error: --mutation requiere especificar --category")
        sys.exit(1)
    
    # Crear analizador de validación
    validator = MutationValidationAnalyzer(args.results_dir)
    
    print("🔍 VALIDACIÓN DEL ANALIZADOR DE MUTACIONES")
    print("=" * 50)
    print(f"MIDI de referencia: {args.midi}")
    print(f"Directorio de resultados: {args.results_dir}")
    
    if args.category and args.mutation:
        # Validar mutación específica
        print(f"Validando mutación específica: {args.category}.{args.mutation}")
        
        result = validator.validate_single_mutation(args.midi, args.category, args.mutation)
        
        if not result:
            print("❌ No se pudo validar la mutación especificada")
            sys.exit(1)
        
        # Mostrar resultados
        print(f"\n📊 RESULTADOS DE VALIDACIÓN:")
        print(f"   Mutación: {result.mutation_category}.{result.mutation_name}")
        print(f"   Verdaderos Positivos: {result.true_positives}")
        print(f"   Falsos Positivos: {result.false_positives}")
        print(f"   Verdaderos Negativos: {result.true_negatives}") 
        print(f"   Falsos Negativos: {result.false_negatives}")
        print(f"   Precisión: {result.precision:.3f}")
        print(f"   Recall: {result.recall:.3f}")
        print(f"   F1-Score: {result.f1_score:.3f}")
        print(f"   Exactitud: {result.accuracy:.3f}")
        print(f"   Cambios esperados: {result.expected_changes}")
        print(f"   Cambios detectados: {result.detected_changes}")
        
    else:
        # Validar todas las mutaciones
        print("Validando todas las mutaciones disponibles...")
        
        validation_result = validator.validate_all_mutations(args.midi)
        
        if not validation_result:
            print("❌ No se pudo ejecutar la validación global")
            sys.exit(1)
        
        # Determinar directorio de salida
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(args.results_dir) / args.midi / "validation"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar reporte de texto
        report_path = output_dir / "validation_report.txt"
        validator.generate_validation_report(validation_result, str(report_path))
        
        # Guardar CSV con resultados
        csv_path = output_dir / "validation_results.csv"
        validator.save_validation_results_csv(validation_result, str(csv_path))
        
        # Generar matriz de confusión si no se deshabilitó
        if not args.no_plots:
            try:
                confusion_path = output_dir / "confusion_matrix.png"
                validator.plot_confusion_matrix(validation_result, str(confusion_path))
            except Exception as e:
                print(f"⚠️ No se pudo generar el gráfico de matriz de confusión: {e}")
        
        # Mostrar resumen
        print(f"\n📊 RESUMEN DE VALIDACIÓN GLOBAL:")
        print(f"   Total mutaciones: {validation_result.total_mutations}")
        print(f"   Precisión global: {validation_result.overall_precision:.3f}")
        print(f"   Recall global: {validation_result.overall_recall:.3f}")
        print(f"   F1-Score global: {validation_result.overall_f1_score:.3f}")
        print(f"   Exactitud global: {validation_result.overall_accuracy:.3f}")
        
        print(f"\n📈 RENDIMIENTO POR CATEGORÍA:")
        for category, metrics in validation_result.category_performance.items():
            print(f"   {category}:")
            print(f"     - Mutaciones: {metrics['mutations_count']}")
            print(f"     - Precisión: {metrics['precision']:.3f}")
            print(f"     - Recall: {metrics['recall']:.3f}")
            print(f"     - F1-Score: {metrics['f1_score']:.3f}")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   - Reporte: {report_path}")
        print(f"   - Datos CSV: {csv_path}")
        if not args.no_plots:
            print(f"   - Matriz de confusión: {output_dir / 'confusion_matrix.png'}")
        
        # Interpretación automática
        print(f"\n💡 INTERPRETACIÓN:")
        if validation_result.overall_f1_score > 0.8:
            print("   ✅ Excelente rendimiento del analizador")
        elif validation_result.overall_f1_score > 0.6:
            print("   ✅ Buen rendimiento del analizador")
        else:
            print("   ⚠️ El analizador necesita mejoras")
        
        if validation_result.overall_precision < 0.7:
            print("   ⚠️ Muchos falsos positivos - el analizador detecta errores donde no los hay")
        
        if validation_result.overall_recall < 0.7:
            print("   ⚠️ Muchos falsos negativos - el analizador pierde errores reales")


if __name__ == "__main__":
    main()
