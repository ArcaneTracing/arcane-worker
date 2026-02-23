"""
Singleton pattern utilities.
Provides thread-safe singleton implementation.
"""
from typing import Type, Optional
import threading


class SingletonMeta(type):
    """
    Thread-safe singleton metaclass.
    
    Usage:
        class MyClass(metaclass=SingletonMeta):
            pass
    """
    _instances: dict = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def get_or_create_singleton[T](
    factory: Type[T],
    instance_var: Optional[T] = None
) -> T:
    """
    Get or create a singleton instance.
    
    Args:
        factory: Class or factory function to create the instance
        instance_var: Optional existing instance variable (for backward compatibility)
        
    Returns:
        Singleton instance
    """
    if instance_var is not None:
        return instance_var
    
    # For classes, use the metaclass if available
    if isinstance(factory, type):
        return factory()
    
    # For factory functions
    return factory()

