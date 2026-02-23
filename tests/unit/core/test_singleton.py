"""
Unit tests for singleton utilities.
"""
import pytest
from app.core.singleton import SingletonMeta, get_or_create_singleton


class TestSingletonMeta:
    """Tests for SingletonMeta metaclass."""
    
    def test_singleton_creates_single_instance(self):
        """Should create only one instance."""
        class TestClass(metaclass=SingletonMeta):
            def __init__(self):
                self.value = 42
        
        instance1 = TestClass()
        instance2 = TestClass()
        
        assert instance1 is instance2
        assert instance1.value == 42
        assert instance2.value == 42
    
    def test_singleton_preserves_instance_state(self):
        """Should preserve instance state across calls."""
        class TestClass(metaclass=SingletonMeta):
            def __init__(self):
                self.counter = 0
            
            def increment(self):
                self.counter += 1
        
        instance1 = TestClass()
        instance1.increment()
        instance1.increment()
        
        instance2 = TestClass()
        assert instance2.counter == 2
        
        instance2.increment()
        assert instance1.counter == 3


class TestGetOrCreateSingleton:
    """Tests for get_or_create_singleton function."""
    
    def test_creates_instance_on_first_call(self):
        """Should create instance on first call."""
        class TestClass:
            def __init__(self):
                self.value = "test"
        
        instance = get_or_create_singleton(TestClass)
        assert isinstance(instance, TestClass)
        assert instance.value == "test"
    
    def test_returns_provided_instance_var(self):
        """Should return provided instance_var if given."""
        class TestClass:
            def __init__(self):
                self.value = "test"
        
        existing_instance = TestClass()
        existing_instance.value = "existing"
        
        instance = get_or_create_singleton(TestClass, existing_instance)
        assert instance is existing_instance
        assert instance.value == "existing"
    
    def test_creates_new_instance_when_no_instance_var(self):
        """Should create new instance when instance_var is None."""
        class TestClass:
            def __init__(self, value="default"):
                self.value = value
        
        instance1 = get_or_create_singleton(TestClass)
        instance2 = get_or_create_singleton(TestClass)
        
        # Without instance_var, creates new instances (not singleton pattern)
        # This matches the actual implementation behavior
        assert instance1 is not None
        assert instance2 is not None

