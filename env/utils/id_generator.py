"""
Unique ID generator

Generates unique IDs in the format <TypeName00001>, supporting multiple data types.

Supports two modes:
- train mode: 6 digits (maximum 999999), typically used for the training set
- test mode: 5 digits (maximum 99999), typically used for the test set

Features:
- Generate unique IDs
- Reset counters
- Persist state to a file
- Load state from a file
"""

import json
from pathlib import Path
from typing import Dict, Optional, Literal
from threading import Lock


class IDGenerator:
    """
    Thread-safe ID generator

    ID format: <TypeName00001> or <TypeName0000001>
    - TypeName: name of the data type
    - train mode: 7 digits, counting from 1, maximum 9999999
    - test mode: 5 digits, counting from 1, maximum 99999
    """
    
    def __init__(self, state_file: Optional[str] = None, mode: Literal["train", "test"] = "test"):
        """
        Initialize the ID generator

        Args:
            state_file: Path to the state file for persisting counter values
            mode: ID generation mode; "train" uses 7 digits (maximum 9999999), "test" uses 5 digits (maximum 99999)
        """
        self._counters: Dict[str, int] = {}  # Counter for each type
        self._lock = Lock()  # Thread lock to ensure thread safety; prevents concurrent updates to the object
        self._state_file = Path(state_file) if state_file else None
        self._mode = mode
        
        # Configure the number of digits and maximum limit according to the mode
        if self._mode == "train":
            self._num_digits = 7
            self._max_limit = 9999999
        else:  # test mode
            self._num_digits = 5
            self._max_limit = 99999
        
        # Load the previous state if a state file is provided and exists
        if self._state_file and self._state_file.exists():
            self.load_state()
    
    def generate_id(self, type_name: str) -> str:
        """
        Generate a unique ID for the specified type

        Args:
            type_name: Name of the data type (e.g., "Location", "Transportation")

        Returns:
            str: Unique ID in the format <TypeName00001> (test mode) or <TypeName000001> (train mode)

        Raises:
            ValueError: If the number of IDs for this type exceeds the maximum limit

        Example:
            >>> generator = IDGenerator(mode="test")
            >>> generator.generate_id("Location")
            '<Location00001>'
            >>> generator = IDGenerator(mode="train")
            >>> generator.generate_id("Location")
            '<Location000001>'
        """
        with self._lock:
            # Hold the lock until the operation completes to prevent concurrent updates
            # Retrieve the current counter for the type, initializing to 0 if absent
            current_count = self._counters.get(type_name, 0)
            
            # Check whether the maximum limit has been reached
            if current_count >= self._max_limit:
                raise ValueError(
                    f"Type '{type_name}' has reached maximum ID limit ({self._max_limit}) in {self._mode} mode. "
                    f"Consider using reset() or a different type name."
                )
            
            # Increment the counter
            new_count = current_count + 1
            self._counters[type_name] = new_count
            
            # Generate the ID string using the digit width for the current mode
            id_string = f"<{type_name}{new_count:0{self._num_digits}d}>"
            
            return id_string
    
    def get_current_count(self, type_name: str) -> int:
        """
        Get the current count for the given type

        Args:
            type_name: Name of the data type

        Returns:
            int: Current count (number of generated IDs)

        Example:
            >>> generator.generate_id("Location")  # Generate the first ID
            >>> generator.get_current_count("Location")
            1
        """
        with self._lock:
            return self._counters.get(type_name, 0)
    
    def reset(self, type_name: Optional[str] = None):
        """
        Reset counters

        Args:
            type_name: Name of the type to reset. If None, reset all types.

        Example:
            >>> generator.reset("Location")  # Reset only the Location type
            >>> generator.reset()           # Reset all types
        """
        with self._lock:
            if type_name is None:
                # Reset all counters
                self._counters.clear()
                print("All ID counters have been reset.")
            else:
                # Reset the counter for the specified type
                if type_name in self._counters:
                    self._counters[type_name] = 0
                    print(f"ID counter for '{type_name}' has been reset.")
                else:
                    print(f"Type '{type_name}' not found in counters.")
    
    def get_all_counts(self) -> Dict[str, int]:
        """
        Get the current counts for all types

        Returns:
            Dict[str, int]: Mapping from type name to count

        Example:
            >>> generator.get_all_counts()
            {'Location': 5, 'Transportation': 3}
        """
        with self._lock:
            return self._counters.copy()
    
    def save_state(self, file_path: Optional[str] = None):
        """
        Save the current counter state to a JSON file

        Args:
            file_path: Target file path. If None, use the path specified during initialization.

        Raises:
            ValueError: If no file path is provided

        Example:
            >>> generator.save_state("counters.json")
        """
        target_file = Path(file_path) if file_path else self._state_file
        
        if not target_file:
            raise ValueError("No file path specified for saving state.")
        
        with self._lock:
            # Ensure the directory exists
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the state to a JSON file
            state_data = {
                "mode": self._mode,
                "counters": self._counters.copy(),
                "metadata": {
                    "total_types": len(self._counters),
                    "total_ids": sum(self._counters.values()),
                    "num_digits": self._num_digits,
                    "max_limit": self._max_limit
                }
            }
            
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=4, ensure_ascii=False)
            
            print(f"ID generator state saved to: {target_file}")
    
    def load_state(self, file_path: Optional[str] = None):
        """
        Load counter state from a JSON file

        Args:
            file_path: File path to load from. If None, use the path specified during initialization.

        Raises:
            ValueError: If no file path is provided, if the file is invalid, or if the mode does not match
            FileNotFoundError: If the specified file does not exist

        Example:
            >>> generator.load_state("counters.json")
        """
        source_file = Path(file_path) if file_path else self._state_file
        
        if not source_file:
            raise ValueError("No file path specified for loading state.")
        
        if not source_file.exists():
            raise FileNotFoundError(f"State file not found: {source_file}")
        
        with self._lock:
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                
                # Check whether the saved mode matches the generator mode (if provided)
                saved_mode = state_data.get("mode")
                if saved_mode and saved_mode != self._mode:
                    print(f"Warning: Loading state from {saved_mode} mode into {self._mode} mode generator. "
                          f"ID format may be inconsistent.")
                
                # Load the counters
                self._counters = state_data.get("counters", {})
                
                # Metadata is available to callers for additional diagnostics if needed
                metadata = state_data.get("metadata", {})
                # print(f"ID generator state loaded from: {source_file}")
                # print(f"Loaded {metadata.get('total_types', 0)} types, "
                #       f"{metadata.get('total_ids', 0)} total IDs")
                
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Invalid state file format: {e}")
    
    def preview_next_ids(self, type_name: str, count: int = 5) -> list[str]:
        """
        Preview the upcoming IDs for the specified type without generating them

        Args:
            type_name: Name of the data type
            count: Number of IDs to preview

        Returns:
            list[str]: List of upcoming IDs

        Example:
            >>> generator.preview_next_ids("Location", 3)
            ['<Location00001>', '<Location00002>', '<Location00003>']  # test mode
            ['<Location000001>', '<Location000002>', '<Location000003>']  # train mode
        """
        with self._lock:
            current_count = self._counters.get(type_name, 0)
            next_ids = []
            
            for i in range(1, count + 1):
                next_count = current_count + i
                if next_count > self._max_limit:
                    break
                next_ids.append(f"<{type_name}{next_count:0{self._num_digits}d}>")
            
            return next_ids
    
    def __repr__(self) -> str:
        """Return a concise string representation of the generator"""
        total_ids = sum(self._counters.values())
        return f"IDGenerator(mode={self._mode}, types={len(self._counters)}, total_ids={total_ids})"


_global_generator = None
_global_lock = Lock()

def get_global_generator(state_file: str = "env/data/runtime/id_history/id_generator_state.json", mode: Literal["train", "test"] = "test") -> IDGenerator:
    """
    Retrieve the global ID generator instance (thread-safe singleton pattern)

    Args:
        state_file: Path to the state file
        mode: ID generation mode; "train" uses 6 digits, "test" uses 5 digits

    Returns:
        IDGenerator: Global ID generator instance
    """
    global _global_generator
    if _global_generator is None:
        with _global_lock:  # Ensure only one thread performs initialization
            if _global_generator is None:  # double-checked locking
                _global_generator = IDGenerator(state_file, mode)
    return _global_generator


# Example usage and test code
if __name__ == "__main__":
    # Basic usage example
    print("=== ID Generator Example ===")
    
    # Create a generator
    generator = IDGenerator()
    
    # Generate some IDs
    print("Generating Location IDs:")
    for i in range(3):
        location_id = generator.generate_id("Location")
        print(f"  {location_id}")
    
    print("\nGenerating Transportation IDs:")
    for i in range(2):
        transport_id = generator.generate_id("Transportation")
        print(f"  {transport_id}")
    
    # View the current state
    print(f"\nCurrent counts: {generator.get_all_counts()}")
    
    # Preview the next IDs
    print(f"\nNext Location IDs: {generator.preview_next_ids('Location', 3)}")
    
    # Test save and load operations
    print("\nTesting save/load functionality...")
    generator.save_state("test_state.json")
    
    # Create a new generator and load the state
    new_generator = IDGenerator()
    new_generator.load_state("test_state.json")
    
    # Continue generating IDs
    next_location_id = new_generator.generate_id("Location")
    print(f"Next Location ID after reload: {next_location_id}")
    
    # Clean up the temporary test file
    import os
    if os.path.exists("test_state.json"):
        os.remove("test_state.json")
        print("Test file cleaned up.")