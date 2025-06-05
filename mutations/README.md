# Mutations Module Structure

The mutations module has been modularized for better code organization and maintainability.

## File Structure

```
mutations/
├── __init__.py           # Package initialization and exports
├── results.py            # MutationResult class
├── category.py           # MutationCategory class  
├── manager.py            # MutationManager class (main orchestrator)
├── controller.py         # Individual mutation functions
├── config.py            # Configuration settings
└── other files...
```

## Classes

### MutationResult (`results.py`)
- Represents the result of applying a single mutation
- Handles mutation execution, success/failure tracking, and error reporting
- Contains the actual mutated data and file paths

### MutationCategory (`category.py`)  
- Groups related mutations together (e.g., pitch errors, tempo errors)
- Manages batch application of mutations within a category
- Provides statistics and filtering for successful/failed mutations

### MutationManager (`manager.py`)
- Main orchestrator that manages all mutation categories
- Initializes all available mutations and their categories
- Provides high-level interface for mutation operations
- Handles summary reporting and export functionality

## Usage Examples

```python
# Import all classes from package root
from mutations import MutationResult, MutationCategory, MutationManager

# Or import specific classes
from mutations.manager import MutationManager
from mutations.results import MutationResult
from mutations.category import MutationCategory

# Create and use mutation manager
manager = MutationManager()
results = manager.apply_all_mutations(excerpt)
manager.print_summary()
```

## Benefits of Modularization

1. **Separation of Concerns**: Each class has a focused responsibility
2. **Maintainability**: Easier to modify individual components without affecting others
3. **Reusability**: Classes can be imported and used independently
4. **Testing**: Individual classes can be unit tested in isolation
5. **Readability**: Smaller, focused files are easier to understand
