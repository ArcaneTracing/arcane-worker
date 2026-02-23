"""
Generic registry pattern for factories and services.
Can be used for any type of registry-based lookup.
"""
from typing import Dict, Optional, Type, Callable, Any


class Registry[T]:
    """
    Generic registry for mapping keys to values.
    
    Can be used for factories, services, or any key-value mapping.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._items: Dict[str, T] = {}
    
    def register(self, key: str, value: T) -> None:
        """
        Register a value with a key.
        
        Args:
            key: The key to register
            value: The value to register
        """
        if key in self._items:
            raise ValueError(f"Key '{key}' is already registered")
        self._items[key] = value
    
    def get(self, key: str) -> Optional[T]:
        """
        Get a value by key.
        
        Args:
            key: The key to look up
            
        Returns:
            The value if found, None otherwise
        """
        return self._items.get(key)
    
    def has(self, key: str) -> bool:
        """
        Check if a key is registered.
        
        Args:
            key: The key to check
            
        Returns:
            True if the key is registered, False otherwise
        """
        return key in self._items
    
    def get_all_keys(self) -> list[str]:
        """
        Get all registered keys.
        
        Returns:
            List of all registered keys
        """
        return list(self._items.keys())
    
    def get_all_items(self) -> Dict[str, T]:
        """
        Get all registered items.
        
        Returns:
            Dictionary of all registered items
        """
        return self._items.copy()


class FactoryRegistry[T](Registry[Callable[..., T]]):
    """
    Registry for factory functions.
    
    Maps keys to factory functions that create instances.
    """
    
    def create(self, key: str, *args, **kwargs) -> Optional[T]:
        """
        Create an instance using the factory function for the given key.
        
        Args:
            key: The key to look up
            *args: Positional arguments to pass to the factory function
            **kwargs: Keyword arguments to pass to the factory function
            
        Returns:
            The created instance, or None if key not found
        """
        factory = self.get(key)
        if factory is None:
            return None
        return factory(*args, **kwargs)
    
    def create_or_raise(self, key: str, *args, **kwargs) -> T:
        """
        Create an instance using the factory function, raising error if not found.
        
        Args:
            key: The key to look up
            *args: Positional arguments to pass to the factory function
            **kwargs: Keyword arguments to pass to the factory function
            
        Returns:
            The created instance
            
        Raises:
            ValueError: If key is not found
        """
        factory = self.get(key)
        if factory is None:
            raise ValueError(f"Unknown key: {key}. Available keys: {', '.join(self.get_all_keys())}")
        return factory(*args, **kwargs)

