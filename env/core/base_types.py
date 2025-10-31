"""
Core base type definitions.

Defines abstract base classes for all data types and tools, providing a unified interface for the system.
These base classes ensure type safety and extensibility.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from dataclasses import dataclass, asdict
import sys
# import asdict
import json
import os
from pathlib import Path
import yaml
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
from env.domains.travel.enums import *


TRAVEL_CONFIG_PATH = "env/domains/travel/travel_config.yaml"
JSON_DECODE_ERROR_OUTPUT = "[ERROR] Model tool call outputs cannot be parsed as JSON."


def get_all_functions() -> List[str]:
    """
    Get the list of all functions.

    Returns:
        List[str]: List of function names
    """
    return [
        "decide", 
        "search",
        "refine",
        "select"
    ]


def parse_model_outputs(model_outputs: str, function: str, task: str) -> Dict[str, Any]:
    """
    Parse the model output string and extract tool-call results.

    Args:
        model_outputs: Raw string output from the LLM
        function: Tool function name, such as "search"
        task: Task name, such as "location"
    """
    try:
        # Attempt direct parsing first
        args = json.loads(model_outputs)
        # If the parsed result is still a string, decode it repeatedly
        depth = 0
        max_depth = 3
        while isinstance(args, str) and depth < max_depth:
            try:
                args = json.loads(args)
            except Exception:
                break
            depth += 1
    except Exception as e:
        # If direct parsing fails, try cleaning the string
        try:
            # Remove potential leading and trailing whitespace and newlines
            cleaned_outputs = model_outputs.strip()
            
            # Handle JSON strings wrapped in a single layer of quotes (e.g. "{\"key\": \"value\"}" -> {"key": "value"})
            # This is the most common failure case
            if (cleaned_outputs.startswith('"') and cleaned_outputs.endswith('"') and 
                cleaned_outputs.count('"') > 2):
                # Remove the outer quotes
                inner = cleaned_outputs[1:-1]
                
                # Check whether the inner segment is valid JSON
                if inner.startswith('{') and inner.endswith('}'):
                    # Handle escaped characters inside the string
                    cleaned_outputs = inner.replace('\\"', '"')
                elif inner.startswith('\\"') and inner.endswith('\\"'):
                    # Handle the "\"...\"" pattern
                    cleaned_outputs = inner[2:-2].replace('\\\\"', '"').replace('\\\\', '\\')
                else:
                    # If the inner section is not standard JSON, leave it for later handling
                    pass
            
            # Handle heavily escaped JSON strings (e.g. "\"{...}\"" -> "{...}")
            elif cleaned_outputs.startswith('"\\"') and cleaned_outputs.endswith('\\""'):
                # Remove the outer quotes and fix escaping
                cleaned_outputs = cleaned_outputs[2:-2]  # Remove "\" and \""
                # Replace \\\" with " and \\\\ with \
                cleaned_outputs = cleaned_outputs.replace('\\\\"', '"').replace('\\\\', '\\')
            
            # If the string does not start with {, try locating the JSON section
            if not cleaned_outputs.startswith('{'):
                # Find the first { and the last }
                start_idx = cleaned_outputs.find('{')
                end_idx = cleaned_outputs.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    cleaned_outputs = cleaned_outputs[start_idx:end_idx+1]
                else:
                    # If a complete JSON structure cannot be found, try to build one
                    # Handles cases such as '\n    "tool_name": "test"\n'
                    if ':' in cleaned_outputs and '"' in cleaned_outputs:
                        # Attempt to wrap it into a full JSON object
                        cleaned_outputs = '{' + cleaned_outputs + '}'
            
            # Handle special end markers (e.g. <｜tool▁call▁end｜>)
            special_endings = ['<｜tool▁call▁end｜>', '<|tool_call_end|>', '<|end|>']
            for ending in special_endings:
                if ending in cleaned_outputs:
                    cleaned_outputs = cleaned_outputs.split(ending)[0]
            
            # Handle JSON without outer quotes but containing escaped segments (e.g. {\"key\": \"value\"})
            if cleaned_outputs.startswith('{') and '\\"' in cleaned_outputs and not cleaned_outputs.startswith('{"'):
                # Replace the inner \\" sequences with "
                cleaned_outputs = cleaned_outputs.replace('\\"', '"')
            
            args = json.loads(cleaned_outputs)
            # If the parsed result is still a string, decode repeatedly
            depth = 0
            max_depth = 3
            while isinstance(args, str) and depth < max_depth:
                try:
                    args = json.loads(args)
                except Exception:
                    break
                depth += 1
        except Exception as e2:
            # Final attempt: handle potential string-format issues
            try:
                # Attempt to resolve potential Unicode escape or encoding issues
                final_attempt = model_outputs.strip()
                
                # Handle potential multi-layer escaping
                if final_attempt.count('\\') > final_attempt.count('"'):
                    # Excessive escaping may exist; try unescaping step by step
                    import re
                    # First replace \\\" with "
                    final_attempt = re.sub(r'\\\\\"', r'\"', final_attempt)
                    # Then replace \" with "
                    final_attempt = re.sub(r'\\\"', r'"', final_attempt)
                    # Replace \\\\ with \\
                    final_attempt = re.sub(r'\\\\\\\\', r'\\\\', final_attempt)
                    # Replace \\ with \
                    final_attempt = re.sub(r'\\\\', r'\\', final_attempt)
                
                # Remove potential outer quotation marks
                if final_attempt.startswith('"') and final_attempt.endswith('"'):
                    final_attempt = final_attempt[1:-1]
                
                # Ensure the result is valid JSON
                if not final_attempt.startswith('{'):
                    start_idx = final_attempt.find('{')
                    if start_idx != -1:
                        end_idx = final_attempt.rfind('}')
                        if end_idx > start_idx:
                            final_attempt = final_attempt[start_idx:end_idx+1]
                    else:
                        # If { is missing but : and " are present, try to repair the incomplete JSON
                        if ':' in final_attempt and '"' in final_attempt:
                            # Check whether the leading {" is missing
                            if not final_attempt.startswith('"') and final_attempt.count('"') >= 2:
                                final_attempt = '{"' + final_attempt
                            elif final_attempt.startswith('"') and not final_attempt.startswith('{"'):
                                final_attempt = '{' + final_attempt
                
                args = json.loads(final_attempt)
                # If the parsed result is still a string, decode repeatedly
                depth = 0
                max_depth = 3
                while isinstance(args, str) and depth < max_depth:
                    try:
                        args = json.loads(args)
                    except Exception:
                        break
                    depth += 1
            except Exception as e3:
                print(f"[ERROR] JSON parsing failed for model_outputs: {repr(model_outputs)}")
                print(f"[ERROR] Original error: {e}")
                print(f"[ERROR] Cleaned parsing error: {e2}")
                print(f"[ERROR] Final attempt error: {e3}")
                return JSON_DECODE_ERROR_OUTPUT

    # Ensure args is a dictionary
    if not isinstance(args, dict):
        print(f"[ERROR] Parsed result is not a dictionary: {type(args)} - {repr(args)}")
        return JSON_DECODE_ERROR_OUTPUT
    
    return {k: str(v) for k, v in args.items()}

def get_all_tasks() -> List[str]:
    """
    Get the list of all tasks.

    Returns:
        List[str]: List of task names
    """
    travel_config_path = Path(TRAVEL_CONFIG_PATH)
    with travel_config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    main_subtasks = raw.get("main_subtasks", {})
    return list(main_subtasks.keys())


class DataType(ABC):
    """
    Abstract base class for all data types.

    Every concrete data type (such as LocationPreference, LocationCandidate, etc.)
    must inherit from this base class and implement its abstract methods.
    """
    
    @abstractmethod
    def get_unique_id(self) -> str:
        """
        Retrieve the unique ID for the data object.

        Returns:
            str: Unique ID in the format <TypeName00001>

        Example:
            >>> preference = LocationPreference(...)
            >>> preference.get_unique_id()
            '<LocationPreference00001>'
        """
        pass
    
    @abstractmethod
    def get_type_name(self) -> str:
        """
        Get the data type name.

        Returns:
            str: Type name, such as "LocationPreference"
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the data object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing all attributes

        Example:
            >>> preference.to_dict()
            {'unique_id': '<LocationPreference00001>', 'category': 'city', ...}
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataType':
        """
        Create a data object from a dictionary.

        Args:
            data: Dictionary containing object attributes

        Returns:
            DataType: Created data object instance
        """
        pass
    
    def __str__(self) -> str:
        """Return the string representation of the object."""
        return f"{self.get_type_name()}({self.get_unique_id()})"
    
    def __repr__(self) -> str:
        """Return the detailed string representation of the object."""
        return f"{self.__class__.__name__}(id='{self.get_unique_id()}')"


class Tool(ABC):
    """
    Abstract base class for all tools.

    Defines the fundamental interface for tools, including name, input/output types, cost, etc.
    Every concrete tool (atomic or composite) must inherit from this base class.
    """
    
    def __init__(
        self, 
        name: str, 
        input_types: List[str], 
        output_type: str, 
        cost: Optional[int] = None,
        description: str = ""
    ):
        """
        Initialize the tool.

        Args:
            name: Tool name
            input_types: List of input types, such as ["LocationPreference"]
            output_type: Output type, such as "LocationCandidate_Raw"
            cost: Tool cost; None indicates not assigned
            description: Tool description
        """
        self.name = name
        self.input_types = input_types
        self.output_type = output_type
        self.cost = cost
        self.description = description
    
    @abstractmethod
    def get_tool_type(self) -> str:
        """
        Get the tool type.

        Returns:
            str: "atomic" or "composite"
        """
        pass
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """
        Get the complete tool definition.

        Returns:
            Dict[str, Any]: Dictionary containing all tool information
        """
        pass
    
    @abstractmethod
    def process_input(self, model_outputs: str) -> Any:
        """
        Process the input string and return the result.

        Args:
            model_outputs: Input string (raw tool-use output from the LLM)

        Returns:
            Any: Processed result
        """
        pass
    
    @abstractmethod
    def process_output(self, tool_output: Any, refinement_level: int) -> str:
        """
        Process the tool output and return a string.

        Args:
            tool_output: Tool execution result
            refinement_level: Refinement level; 0 means initial output, 1 means first refinement, and so on

        Returns:
            str: Processed string
        """
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the tool functionality (abstract method).

        Concrete tool implementations must define how to handle inputs and produce outputs.
        During data generation, this method is primarily used for type validation.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Any: Tool execution result
        """
        pass
    
    def validate_inputs(self, inputs: List[Any]) -> bool:
        """
        Validate whether the input types meet the requirements.

        Args:
            inputs: List of input parameters

        Returns:
            bool: True if the input types are correct
        """
        if len(inputs) != len(self.input_types):
            return False
        
        return True
    
    def set_cost(self, cost: int):
        """
        Set the tool cost.

        Args:
            cost: Tool cost value
        """
        if cost <= 0:
            raise ValueError("[ERROR] Tool cost must be positive")
        self.cost = cost
    
    def has_cost(self) -> bool:
        """
        Check whether the tool has an assigned cost.

        Returns:
            bool: True if a cost has been assigned
        """
        return self.cost is not None
    
    def generate_interface(self, provide_composite_concept: bool = False) -> Dict[str, Any]:
        """
        Generate the standard tool calling interface.

        Returns:
            Dict[str, Any]: Standard-format tool calling interface
        """
        # Generate properties for the parameters
        properties = {}
        for input_type in self.input_types:
            # Generate the description based on the type
            if input_type.endswith("Preference"):
                properties[input_type] = {
                    "type": "string",
                    "description": f"A {input_type} object identifier representing user preferences for {input_type.replace('Preference', '').lower()} selection"
                }
            elif input_type.endswith("Candidate_Raw"):
                properties[input_type] = {
                    "type": "string", 
                    "description": f"A {input_type} object identifier representing raw search results for {input_type.split('_')[0].lower()}"
                }
            elif "Candidate_L" in input_type and input_type.endswith("_Available"):
                properties[input_type] = {
                    "type": "string",
                    "description": f"A {input_type} object identifier representing candidates refined to L1 level (availability checked)"
                }
            elif "Candidate_L" in input_type and input_type.endswith("_Located"):
                properties[input_type] = {
                    "type": "string", 
                    "description": f"A {input_type} object identifier representing candidates refined to L2 level (location analyzed)"
                }
            elif "Candidate_L" in input_type and input_type.endswith("_Reviewed"):
                properties[input_type] = {
                    "type": "string",
                    "description": f"A {input_type} object identifier representing candidates refined to L3 level (reputation reviewed)"
                }
            elif "Candidate_L" in input_type and input_type.endswith("_Verified"):
                properties[input_type] = {
                    "type": "string",
                    "description": f"A {input_type} object identifier representing candidates refined to L4 level (restrictions verified)"
                }
            elif "Candidate_L" in input_type and input_type.endswith("_Finalized"):
                properties[input_type] = {
                    "type": "string",
                    "description": f"A {input_type} object identifier representing candidates refined to L5 level (cost-effectiveness finalized)"
                }
            elif input_type == "TimeInfo":
                properties[input_type] = {
                    "type": "string",
                    "description": "A TimeInfo object identifier representing a time period (not specific time points) for travel planning. The task execution is independent of actual time values"
                }
            elif input_type.startswith("HomeLocation"):
                properties[input_type] = {
                    "type": "string",
                    "description": f"A {input_type} object identifier representing the user's home location information"
                }
            # Handle enum types — this is the key change
            elif input_type in ["LocationCategory", "LocationTier", "LocationStyle", "LocationFeaturePackage",
                            "TransportationCategory", "TransportationTier", "TransportationStyle", "TransportationFeaturePackage",
                            "AccommodationCategory", "AccommodationTier", "AccommodationStyle", "AccommodationFeaturePackage",
                            "AttractionCategory", "AttractionTier", "AttractionStyle", "AttractionFeaturePackage",
                            "DiningCategory", "DiningTier", "DiningStyle", "DiningFeaturePackage",
                            "ShoppingCategory", "ShoppingTier", "ShoppingStyle", "ShoppingFeaturePackage"]:
                
                # Get the corresponding enum class
                from env.domains.travel.enums import ENUM_DIRECT_MAPPINGS
                enum_class = ENUM_DIRECT_MAPPINGS[input_type]
                
                # Extract all enum values
                enum_values = [e.value for e in enum_class]
                
                # Extract the base type (remove Category/Tier/Style/FeaturePackage suffix)
                base_type = input_type.replace("Category", "").replace("Tier", "").replace("Style", "").replace("FeaturePackage", "").lower()
                
                if input_type.endswith("Category"):
                    properties[input_type] = {
                        "type": "string",
                        "enum": enum_values,  # Add enum value options
                        "description": f"A {base_type} category enumeration value specifying the main type/category for {base_type} selection. Available options: {', '.join(enum_values)}"
                    }
                elif input_type.endswith("Tier"):
                    properties[input_type] = {
                        "type": "string",
                        "enum": enum_values,  # Add enum value options
                        "description": f"A {base_type} tier enumeration value specifying the quality/price level for {base_type} selection. Available options: {', '.join(enum_values)}"
                    }
                elif input_type.endswith("Style"):
                    properties[input_type] = {
                        "type": "string",
                        "enum": enum_values,  # Add enum value options
                        "description": f"A {base_type} style enumeration value specifying the preferred style/approach for {base_type} selection. Available options: {', '.join(enum_values)}"
                    }
                elif input_type.endswith("FeaturePackage"):
                    properties[input_type] = {
                        "type": "string",
                        "enum": enum_values,  # Add enum value options
                        "description": f"A {base_type} feature package enumeration value specifying additional features/services for {base_type} selection. Available options: {', '.join(enum_values)}"
                    }
            else:
                # Default handling
                properties[input_type] = {
                    "type": "string",
                    "description": f"A {input_type} object identifier"
                }
            
        # Append descriptions about type/components/cost/output type
        type_extra = ""
        component_extra = ""
        if provide_composite_concept:
            tool_type = self.get_tool_type()
            type_extra = f" Tool type: {tool_type}."
            if tool_type == "composite":
                try:
                    component_names = getattr(self, "component_tool_names", [])
                    if component_names:
                        joined = ", ".join(component_names)
                        component_extra = f" Component tools: {joined}. These are atomic tools composing this composite tool."
                except Exception:
                    pass
        cost_description = f" This tool has a cost of {self.cost} units." if self.cost is not None else ""
        output_type_description = f" The output type of this tool is {self.output_type}."
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": type_extra + self.description + component_extra + cost_description + output_type_description,
                "parameters": {
                    "type": "object", 
                    "properties": properties,
                    "required": list(self.input_types)
                }
            }
        }
    
    def __str__(self) -> str:
        """Return the string representation of the tool."""
        cost_str = f", cost={self.cost}" if self.cost is not None else ""
        return f"{self.name}({self.get_tool_type()}{cost_str})"
    
    def __repr__(self) -> str:
        """Return the detailed string representation of the tool."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"inputs={self.input_types}, output='{self.output_type}', "
                f"cost={self.cost})")
        
        
@dataclass
class BasePreference(DataType, ABC):
    """Abstract base class for all preference types."""
    unique_id: str
    category: str          # Concrete values are defined by subclasses
    tier: str              # Concrete values are defined by subclasses
    style: str             # Concrete values are defined by subclasses
    feature_package: str   # Concrete values are defined by subclasses
    
    def get_unique_id(self) -> str:
        return self.unique_id
    
    @abstractmethod
    def get_type_name(self) -> str:
        """Subclasses must implement the concrete type name."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BasePreference':
        return cls(**data)
    
    @classmethod
    def create(cls, category: str, tier: str, style: str, feature_package: str) -> 'BasePreference':
        """Convenience factory that auto-generates the unique_id."""
        from env.utils.id_generator import get_global_generator
        id_gen = get_global_generator()
        type_name = cls.__name__
        unique_id = id_gen.generate_id(type_name)
        return cls(unique_id, category, tier, style, feature_package)


class AtomicTool(Tool):
    """
    Atomic tool class.

    Atomic tools are the smallest indivisible tool units, each performing a specific function.
    """
    
    def __init__(
        self, 
        name: str, 
        input_types: List[str], 
        output_type: str, 
        cost: Optional[int] = None,
        description: str = ""
    ):
        """
        Initialize an atomic tool.

        Args:
            name: Tool name
            input_types: List of input types
            output_type: Output type
            cost: Tool cost
            description: Tool description
        """
        super().__init__(name, input_types, output_type, cost, description)
    
    def get_tool_type(self) -> str:
        """Return the tool type."""
        return "atomic"
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """
        Retrieve the full definition of the atomic tool.

        Returns:
            Dict[str, Any]: Definition dictionary of the tool
        """
        return {
            "name": self.name,
            "type": "atomic",
            "input_types": self.input_types,
            "output_type": self.output_type,
            "cost": self.cost,
            "description": self.description
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Export as a serializable dictionary.
        """
        return {
            "name": self.name,
            "type": "atomic",
            "input_types": self.input_types,
            "output_type": self.output_type,
            "cost": self.cost,
            "description": self.description
        }
        
    def process_input(self, model_outputs: str) -> Any:
        """
        Process the input string and return the result.

        Args:
            model_outputs: Input string (raw tool-use output from the LLM)

        Returns:
            Any: Processed result
        """
        # Identify the tool type
        self_task = self.get_self_task()
        self_function = ""
        for function in get_all_functions():
            if function.lower() in self.name.lower():
                self_function = function
                break
            
        if not self_task:
            return {"error_message": f"[ERROR] Unknown task detected. Available tasks: {get_all_tasks()}"}
        
        # Parse the LLM output into structured data
        parsed_outputs = parse_model_outputs(model_outputs, self_function, self_task)
        # When parsing fails, parse_model_outputs returns an error string
        if isinstance(parsed_outputs, str):
            return {"error_message": parsed_outputs}
        
        # Construct input objects based on different types
        if self_function == "decide":
            """
            Expected output format of parse_model_outputs in this case:
            parsed_outputs = {
                "LocationCategory": <category_string_id>,
                "LocationTier": <tier_string_id>,
                "LocationStyle": <style_string_id>,
                "LocationFeaturePackage": <feature_package_string_id>,
            }
            If the task is not location, then it becomes:
            {
                "{Task}Category": <category_string_id>,
                "{Task}Tier": <tier_string_id>,
                "{Task}Style": <style_string_id>,
                "{Task}FeaturePackage": <feature_package_string_id>,
                "LocationPreference": <location_preference_string_id>
            }
            Goal: convert into
            {
                "LocationCategory": <LocationCategory Enum Object>,
                "LocationTier": <LocationTier Enum Object>,
                "LocationStyle": <LocationStyle Enum Object>,
                "LocationFeaturePackage": <LocationFeaturePackage Enum Object>
            }
            """
            # Retrieve enum classes from ENUM_DIRECT_MAPPINGS
            from env.domains.travel.enums import ENUM_DIRECT_MAPPINGS
            
            result = {}
            for type_name in self.input_types:
                # Unified existence check to avoid KeyError
                if type_name not in parsed_outputs:
                    return {"error_message": f"[ERROR] Missing input type: {type_name}"}
                
                if type_name != "LocationPreference":  # Then it must be an enum type
                    # First validate whether the enum type exists
                    if type_name not in ENUM_DIRECT_MAPPINGS:
                        return {"error_message": f"[ERROR] Unknown enum type: {type_name}."}
                    enum_class = ENUM_DIRECT_MAPPINGS[type_name]
                    # Then check whether the enum value exists
                    if parsed_outputs[type_name] not in [member.value for member in enum_class]:
                        return {"error_message": f"[ERROR] Invalid enum value for {type_name}: {parsed_outputs[type_name]}. Available options: {[member.value for member in enum_class]}"}
                
                if self_task.lower() == "location":
                    enum_class = ENUM_DIRECT_MAPPINGS[type_name]
                    # Create the enum object using the string ID
                    enum_obj = enum_class(parsed_outputs[type_name])
                    result[type_name] = enum_obj
                    
                else:
                    if type_name == "LocationPreference":
                        # Handle LocationPreference separately
                        from env.core.data_types import DATA_TYPE_REGISTRY
                        LocationPreference_class = DATA_TYPE_REGISTRY["LocationPreference"]
                        location_preference_obj = LocationPreference_class(
                            unique_id=parsed_outputs[type_name],
                            category=parsed_outputs.get("LocationCategory", ""),
                            tier=parsed_outputs.get("LocationTier", ""),
                            style=parsed_outputs.get("LocationStyle", ""),
                            feature_package=parsed_outputs.get("LocationFeaturePackage", "")
                        )
                        result[type_name] = location_preference_obj
                    else:
                        enum_class = ENUM_DIRECT_MAPPINGS[type_name]
                        enum_obj = enum_class(parsed_outputs[type_name])
                        result[type_name] = enum_obj
            
            return result
            
        elif self_function == "search":
            """
            Expected output format of parse_model_outputs in this case:
            For the location task:
            {
                "LocationPreference": <preference_string_id>,
                "TimeInfo": <time_info_string_id>
            }
            For other tasks:
            {
                "{Task}Preference": <preference_string_id>,
                "LocationPreference": <location_preference_string_id>,
                "TimeInfo": <time_info_string_id>
            }
            """
            from env.core.data_types import DATA_TYPE_REGISTRY

            result = {}
            for input_type in self.input_types:
                if input_type in DATA_TYPE_REGISTRY:
                    # Retrieve the class from DATA_TYPE_REGISTRY
                    DataClass = DATA_TYPE_REGISTRY[input_type]
                    if input_type not in parsed_outputs:
                        return {"error_message": f"[ERROR] Missing input type: {input_type}"}
                    string_id = parsed_outputs[input_type]
                    # print("[RUNTIME DEBUG] Processing input type:", input_type, "with string_id:", string_id)
                    if input_type.endswith("Preference"):
                        result[input_type] = DataClass(
                            unique_id=string_id,
                            category="",
                            tier="",
                            style="",
                            feature_package=""
                        )
                        # print(f"[RUNTIME DEBUG] Created {input_type} object:", result[input_type].unique_id)
                    elif input_type == "TimeInfo":
                        result[input_type] = DataClass(
                            unique_id=string_id,
                            departure_time_id="",
                            return_time_id=""
                        )
                    else:
                        return {"error_message": f"[ERROR] Unexpected input type for {self_function} type function: {input_type}"}
                else:
                    return {"error_message": f"[ERROR] Unknown input type: {input_type}"}
            
            return result
                
        elif self_function == "refine":
            """
            Expected output format of parse_model_outputs in this case:
            For Step 1:
            {
                "{Task}Candidate_Raw": <candidate_raw_string_id>,
                "TimeInfo": <time_info_string_id>
            }
            For Step N (N>1):
            {
                "{Task}Candidate_L{N-1}": <candidate_string_id>
            }
            """
            from env.core.data_types import DATA_TYPE_REGISTRY

            result = {}
            for input_type in self.input_types:
                if input_type in DATA_TYPE_REGISTRY:
                    DataClass = DATA_TYPE_REGISTRY[input_type]
                    # Check existence to avoid KeyError
                    if input_type not in parsed_outputs:
                        return {"error_message": f"[ERROR] Missing input type: {input_type}"}
                    string_id = parsed_outputs[input_type]
                    
                    if input_type.endswith("Candidate_Raw"):
                        # Raw classes have no previous candidate info; only unique_id with empty preference
                        result[input_type] = DataClass(
                            unique_id=string_id,
                            preference_id="",
                            preference=None
                        )
                    elif "Candidate_L" in input_type:
                        # L1~L5 candidate classes include previous candidate fields
                        result[input_type] = DataClass(
                            unique_id=string_id,
                            previous_candidate_id=string_id,
                            previous_candidate=None
                        )
                    elif input_type == "TimeInfo":
                        result[input_type] = DataClass(
                            unique_id=string_id,
                            departure_time_id="",
                            return_time_id=""
                        )
                    else:
                        return {"error_message": f"[ERROR] Unexpected input type for {self_function} type function: {input_type}"}
                else:
                    return {"error_message": f"[ERROR] Unknown input type: {input_type}"}
            
            return result
        
        elif self_function == "select":
            """
            Expected output format of parse_model_outputs in this case:
            {
                "{Task}Candidate_L{refinement_level}": <candidate_string_id>
            }
            """
            from env.core.data_types import DATA_TYPE_REGISTRY

            result = {}
            for input_type in self.input_types:
                if input_type in DATA_TYPE_REGISTRY:
                    DataClass = DATA_TYPE_REGISTRY[input_type]
                    # Check existence to avoid KeyError
                    if input_type not in parsed_outputs:
                        return {"error_message": f"[ERROR] Missing input type: {input_type}"}
                    string_id = parsed_outputs[input_type]
                    
                    if "Candidate" in input_type:
                        if input_type.endswith("Candidate_Raw"):
                            result[input_type] = DataClass(
                                unique_id=string_id,
                                preference_id="",
                                preference=None
                            )
                        else:
                            # L1~L5 candidate classes
                            result[input_type] = DataClass(
                                unique_id=string_id,
                                previous_candidate_id="",
                                previous_candidate=None
                            )
                    else:
                        return {"error_message": f"[ERROR] Unexpected input type for {self_function} type function: {input_type}"}
                else:
                    return {"error_message": f"[ERROR] Undefined input type: {input_type}."}
            
            return result     
        
        else:
            return {"error_message": f"[ERROR] Wrong tool call detected. Please input the exact tool name."}
        
        
    def execute(self, input_data: Dict[str, Any], available_tools: Dict[str, Tool]) -> Any:
        """
        Execute the atomic tool.

        Args:
            input_data: Input data dictionary in the format {"TypeName": TypeObject, ...}
            available_tools: Dictionary of currently available atomic tools (virtual input to align with the CompositeTool interface)

        Returns:
            Any: Tool execution result (object matching the output type)
        """
        # Verify that the input is valid
        if "error_message" in input_data:
            return input_data["error_message"]

        # Preliminary parameter validation
        expected_keys = set(self.input_types)
        actual_keys = set(input_data.keys())
        if expected_keys != actual_keys:
            return f"[ERROR] Parameter mismatch. Expected: {expected_keys}, Got: {actual_keys}"
        
        # Validate inputs
        if not self.validate_inputs(input_data):
            return f"[ERROR] Input validation failed for tool {self.name}"

        # Begin execution
        from env.core.data_types import DATA_TYPE_REGISTRY
        from env.utils.id_generator import get_global_generator

        # Decide Preference Tool
        task = self.get_self_task()
        if "Decide" in self.name:
            PreferenceClass = DATA_TYPE_REGISTRY[self.output_type]
            
            from env.settings import load_config

            search_space_path = load_config().paths.search_space_path
            if task == "location":
                unique_id = get_id_from_search_space(
                    search_space_path=search_space_path,
                    task=task,
                    category=str(input_data[f"{task.capitalize()}Category"].value),
                    tier=str(input_data[f"{task.capitalize()}Tier"].value),
                    style=str(input_data[f"{task.capitalize()}Style"].value),
                    feature_package=str(input_data[f"{task.capitalize()}FeaturePackage"].value),
                    LocationPreferenceID=None
                )
                if unique_id.startswith("[ERROR]"):
                    return unique_id
            else:
                if "LocationPreference" not in input_data:
                    return f"[ERROR] Missing LocationPreference input for decide tool {self.name}"
                LocationPreferenceID = input_data["LocationPreference"].unique_id
                # Normalize IDs without angle brackets from the model to the <...> format
                try:
                    if isinstance(LocationPreferenceID, str):
                        raw_id = LocationPreferenceID.strip()
                        if not (raw_id.startswith("<") and raw_id.endswith((">"))):
                            LocationPreferenceID = f"<{raw_id.strip('<>')}>"
                except Exception:
                    pass
                unique_id = get_id_from_search_space(
                    search_space_path=search_space_path,
                    task=task,
                    category=str(input_data[f"{task.capitalize()}Category"].value),
                    tier=str(input_data[f"{task.capitalize()}Tier"].value),
                    style=str(input_data[f"{task.capitalize()}Style"].value),
                    feature_package=str(input_data[f"{task.capitalize()}FeaturePackage"].value),
                    LocationPreferenceID=LocationPreferenceID
                )
                if unique_id.startswith("[ERROR]"):
                    return unique_id
            return PreferenceClass(
                unique_id=unique_id,
                category=str(input_data[f"{task.capitalize()}Category"].value),
                tier=str(input_data[f"{task.capitalize()}Tier"].value),
                style=str(input_data[f"{task.capitalize()}Style"].value),
                feature_package=str(input_data[f"{task.capitalize()}FeaturePackage"].value)
            )

        # Search Tool
        elif "Search" in self.name:
            # Every task requires the LocationPreference
            if "LocationPreference" not in input_data:
                return f"[ERROR] Missing LocationPreference input for search tool {self.name}"
            OutputClass = DATA_TYPE_REGISTRY[self.output_type]
            if task == "location":
                required_pref_type = "LocationPreference"
            else:
                required_pref_type = f"{task.capitalize()}Preference"
            prev_obj = input_data[required_pref_type]
            return OutputClass(
                unique_id=prev_obj.unique_id,
                preference_id=prev_obj.unique_id,
                preference=None
            )

        # Refinement Tool
        elif "Refinement" in self.name:
            OutputClass = DATA_TYPE_REGISTRY[self.output_type]
            required_data_type = None
            for input_type in input_data.keys():
                if "Candidate" in input_type:
                    required_data_type = input_type
                    break
            if not required_data_type:
                return f"[ERROR] Missing candidate input for refinement tool {self.name}"
            return OutputClass(
                unique_id=input_data[required_data_type].unique_id,
                previous_candidate_id=input_data[required_data_type].unique_id,
                previous_candidate=None
            )

        # Select Tool
        elif "Select" in self.name:
            OutputClass = DATA_TYPE_REGISTRY[self.output_type]
            if len(input_data) != 1:
                return f"[ERROR] Select tool {self.name} expects exactly one input"
            required_data_type = list(input_data.keys())[0]
            return OutputClass(
                unique_id=input_data[required_data_type].unique_id,
                previous_candidate_id=input_data[required_data_type].unique_id,
                previous_candidate=None
            )

        else:
            return f"Unknown tool type: {self.name}"
        
        
    def process_output(self, tool_output: Any, refinement_level: int, available_tools: Dict[str, Tool]) -> str:
        """
        Process the tool output and return a string.

        Args:
            tool_output: Tool execution result
            refinement_level: Refinement level; 0 means initial output, 1 means first refinement, and so on
            available_tools: Dictionary of currently available atomic tools (virtual input to align with the CompositeTool interface)

        Returns:
            str: Processed string

        Formatting rules:
            1. Decide tools:
                "Get {unique_id} {task} Preference package. (Type: {self.output_type})"
            2. Search tools:
                refinement_level != 0: "Get raw candidate set: <{Task}CandidateSet{digits}> (unfiltered, Type: {self.output_type})"
                refinement_level == 0: "Get final candidate set: <{Task}CandidateSet{digits}> (unfiltered, Type: {self.output_type})"
            3. Refinement tools:
                refinement_level != current_step: "Filtered by {current_filtering_dimension}. Get filtered candidate set: <{Task}CandidateSet{digits}> (filtered by {past_filtered_dimensions}, Type: {self.output_type})"
                refinement_level == current_step: "Filtered by {current_filtering_dimension}. Get final candidate set: <{Task}CandidateSet{digits}> (filtered by {past_filtered_dimensions}, Type: {self.output_type})"
            4. Select tools:
                "Selected final {task} candidate: <{Task}Candidate{digits}> (Type: {self.output_type})"

            Note: past_filtered_dimensions includes the current filtering dimension.
        """
        # Start with error checks
        if tool_output is None:
            return "[ERROR] Tool output is None."
        elif isinstance(tool_output, str) and tool_output.startswith("[ERROR]"):
            return tool_output
        
        # Check whether tool_output is an object with an attribute named "unique_id"
        if hasattr(tool_output, "unique_id"):
            # If so, assign the value of that "unique_id" attribute to uid
            uid = tool_output.unique_id
        # If the previous condition is not met, check whether tool_output itself is a string
        elif isinstance(tool_output, str):
            # If it is, assign the string directly to uid
            uid = tool_output
        # If neither of the above conditions is true
        else:
            # Log an error message indicating that we encountered an unexpected output type
            print(f"[ERROR] Unexpected tool output type for {self.name} tool:", type(tool_output))
            # As a last resort, cast the entire tool_output to a string and assign it to uid
            uid = str(tool_output)

        # --- Step 2: Decide how to format the output string based on the tool name (self.name) ---
        final_output = ""
        # Check whether the current tool name contains the word "Decide"
        task = self.get_self_task()
        if "Decide" in self.name:
            # If it is a "Decide" tool, create a basic description
            final_output = f"Get {uid} (Type: {self.output_type}). This is a unique id for user's {task} Preference package. "

        # If the tool name does not contain "Decide", check whether it contains "Search"
        elif "Search" in self.name:
            
            # If it is a "Search" tool, return a simple message indicating that a candidate set was found
            id_number = get_all_digits(uid)
            processed_uid = f"<{task.capitalize()}CandidateSet{id_number}>"
            if self.is_last_refinement_step(refinement_level):
                # If this is the final refinement step, return a message indicating that the final candidate set was found
                final_output = f"Get final candidate set: {processed_uid} (unfiltered, Type: {self.output_type})"
            else:
                final_output = f"Get raw candidate set: {processed_uid} (unfiltered, Type: {self.output_type})"

        # If the tool name also lacks "Search", check whether it contains "Refinement"
        elif "Refinement" in self.name:
            
            filter_dimension_map = {
                1: "availability",
                2: "location",
                3: "reputation",
                4: "restrictions",
                5: "cost-effectiveness"
            }
            
            current_step = int(get_all_digits(self.output_type))
            current_filtering_dimension = filter_dimension_map[current_step]
            past_filtered_dimensions = ""
            for step in range(1, current_step + 1):
                past_filtered_dimensions += filter_dimension_map[step] + ", "
            id_number = get_all_digits(uid)
            processed_uid = f"<{task.capitalize()}CandidateSet{id_number}>"
            if self.is_last_refinement_step(refinement_level):
                # If this is the final refinement step, return a message indicating that the final candidate set was found
                final_output = f"Filtered by {current_filtering_dimension}. Get final candidate set: {processed_uid} (Already filtered by {past_filtered_dimensions[:-2]}, Type: {self.output_type})"
            else:
                final_output = f"Filtered by {current_filtering_dimension}. Get filtered candidate set: {processed_uid} (Already filtered by {past_filtered_dimensions[:-2]}, Type: {self.output_type})"

        # If the tool name also lacks "Refinement", check whether it contains "Select"
        elif "Select" in self.name:
            # If it is a "Select" tool, return a message indicating that the final candidate has been chosen
            processed_uid = f"<{task.capitalize()}Candidate{get_all_digits(uid)}>"
            final_output = f"Selected final {task} candidate: {processed_uid} (Type: {self.output_type})."

        # If none of the specific tool name patterns match
        else:
            raise ValueError(f"[ERROR] Unknown tool name for processing output: {self.name}")

        # final_output += ("Cost: " + (str(self.cost) if self.cost is not None else "N/A"))
        # if "N/A" in final_output:
        #     raise ValueError(f"[ERROR] Tool cost not set for tool: {self.name}")
        from env.run import ID_LENGTH
        if get_all_digits(final_output) == "":
            return "[ERROR] No ID found in the input, please check the tool input."
        elif len(get_all_digits(uid)) != ID_LENGTH:
            return "[ERROR] Invalid ID length in the input, please check the tool input."
        return final_output
    
            
    def is_last_refinement_step(self, refinement_level: int) -> bool:
        """
        Check whether this is the final refinement step
        
        Args:
            refinement_level: Current refinement level
            
        Returns:
            bool: True if this is the final step
        """
        if "Select" in self.name or "Decide" in self.name:
            raise ValueError("[ERROR] is_last_refinement_step should not be called for Select or Decide tools.")
        
        elif refinement_level == 0 and "Search" in self.name:
            # This tool is a Search tool and does not refine
            return True
        
        elif "Refinement" in self.name and refinement_level == int(get_all_digits(self.output_type)):
            # This tool is a Refinement tool and it is the final step
            return True
        
        # else:
        #     raise ValueError(f"[ERROR] Cannot determine if last refinement step for tool: {self.name}")
    
        return False


    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate whether the input types meet the requirements
        
        Args:
            inputs: Input parameter dict = {
                "TypeName": TypeObject,
                ...
            }
            
        Returns:
            bool: True if the input types are correct
        """
        if len(inputs) != len(self.input_types):
            return False
        
        from env.core.data_types import DATA_TYPE_REGISTRY
        from env.domains.travel.enums import ENUM_DIRECT_MAPPINGS
        
        # Check that each input type really matches the key
        for input_typename, input_obj in inputs.items():
            if input_typename not in self.input_types:
                return False
            
            # Check whether the type matches
            if input_typename in DATA_TYPE_REGISTRY:
                expected_class = DATA_TYPE_REGISTRY[input_typename]
                if not isinstance(input_obj, expected_class):
                    print(f"[ERROR] Input type mismatch for {input_typename}: expected {expected_class}, got {type(input_obj)}  ")
                    return False
            elif input_typename in ENUM_DIRECT_MAPPINGS:
                expected_enum = ENUM_DIRECT_MAPPINGS[input_typename]
                if not isinstance(input_obj, expected_enum):
                    print(f"[ERROR] Input enum type mismatch for {input_typename}: expected {expected_enum}, got {type(input_obj)}")
                    return False
            else:
                print(f"[ERROR] Unknown input type for validation (Not Self-defined Datatype nor Self-defined Enum type): {input_typename}")
                return False
        
        return True
    
    
    def get_self_task(self) -> str:
        """
        Extract the task name from the tool name
        
        Returns:
            str: Task name, such as "location" or "transportation"
        """
        for task in get_all_tasks():
            if task.lower() in self.name.lower():
                return task
        raise ValueError(f"[ERROR] Cannot determine task from tool name: {self.name}")
        
        
def get_all_digits(source_str: str) -> str:
    """Extract all digits from a string and return them concatenated"""
    for char in source_str:
        if not char.isdigit():
            source_str = source_str.replace(char, "")
    return source_str
    
def get_id_from_search_space(
    search_space_path: str,
    task: str,
    category: str,
    tier: str,
    style: str,
    feature_package: str,
    LocationPreferenceID: str = None
) -> str:
    """
    Retrieve the ID for the specified type from the search space file
    
    Args:
        search_space_path: Path to the search space file
        task: Task name, such as "location" or "transportation"
        category: Category enum value
        tier: Tier enum value
        style: Style enum value
        feature_package: Feature package enum value
        
    Returns:
        str: Retrieved ID
        
    Raises:
        ValueError: If the specified type ID cannot be found
    """
    import json
    with open(search_space_path, 'r', encoding='utf-8') as f:
        search_space = json.load(f)

    # Build the preference tuple in the same order as generate_search_space.py
    # generate_search_space.py uses sorted(dimensions.keys())
    # Alphabetical order: category, feature_package, style, tier
    preference_values = {
        "category": category,
        "feature_package": feature_package, 
        "style": style,
        "tier": tier
    }
    
    # Construct the tuple in alphabetical order (consistent with the sorted order in generate_search_space.py)
    sorted_dimension_names = sorted(preference_values.keys())  # ["category", "feature_package", "style", "tier"]
    preference = tuple(preference_values[dim] for dim in sorted_dimension_names)

    if task == "location":
        unique_id = search_space["locations"].get(str(preference), None)
        if not unique_id:
            return f"[ERROR] Location preference not found: {str(preference)}. Please check the input format and use the exact information given."
    
    else:
        if "location_information" in search_space and LocationPreferenceID:
            location_information = search_space["location_information"]
            if location_information and LocationPreferenceID in location_information:
                LocationInformationID = location_information[LocationPreferenceID]
                if LocationInformationID and task in LocationInformationID:
                    LocationTaskInformationID = LocationInformationID[task]
                    if LocationTaskInformationID and str(preference) in LocationTaskInformationID:
                        unique_id = LocationTaskInformationID[str(preference)]
                    else:
                        return f"[ERROR] No {str(preference)} key founded in search space. Please check the input format and use the exact information given."
                else:
                    return f"[ERROR] No {task} key founded in search space. Please check the input format and use the exact information given."
            else:
                return f"[ERROR] No 'location_information' or '{LocationPreferenceID}' key found in search_space. Please check the input format and use the exact information provided. Make sure your LocationPreference ID is surrounded by '<>'."
        else:
            return f"[ERROR] No location_information key founded in search space or LocationPreferenceID is None. Please check the input format and use the exact information given."

    return unique_id


# Helper functions related to tools
def create_tool_from_dict(tool_data: Dict[str, Any]) -> Tool:
    """
    Create a tool instance from a dictionary
    
    Args:
        tool_data: Dictionary containing tool information
        
    Returns:
        Tool: The created tool instance
        
    Raises:
        ValueError: If the tool type is unknown
    """
    tool_type = tool_data.get("type")
    
    if tool_type == "atomic":
        return AtomicTool(
            name=tool_data["name"],
            input_types=tool_data["input_types"],
            output_type=tool_data["output_type"],
            cost=tool_data.get("cost"),
            description=tool_data.get("description", "")
        )
    elif tool_type == "composite":
        from env.domains.travel.composite_tools import CompositeTool
        return CompositeTool(
            # name=tool_data["name"],
            component_tool_names=tool_data["component_tool_names"],
            refinement_level=tool_data["refinement_level"],
            # input_types=tool_data["input_types"],
            # output_type=tool_data["output_type"],
            cost=tool_data.get("cost"),
            # description=tool_data.get("description", "")
        )
    else:
        raise ValueError(f"[ERROR] Unknown tool type: {tool_type}")
    
if __name__ == "__main__":
    import json
    from env.domains.travel.atomic_tools import (
        Decide_Location_Preference,
        Search_Location_Candidates,
        Location_Refinement_Step1,
        Location_Refinement_Step2,
        Location_Refinement_Step3,
        Location_Refinement_Step4,
        Location_Refinement_Step5,
        Decide_Transportation_Preference,
        Search_Transportation_Candidates,
        Transportation_Refinement_Step1,
        Transportation_Refinement_Step2,
        Transportation_Refinement_Step3,
        Transportation_Refinement_Step4,
        Transportation_Refinement_Step5,
        Decide_Accommodation_Preference,
        Search_Accommodation_Candidates,
        Accommodation_Refinement_Step1,
        Accommodation_Refinement_Step2,
        Accommodation_Refinement_Step3,
        Accommodation_Refinement_Step4,
        Accommodation_Refinement_Step5,
        Decide_Attraction_Preference,
        Search_Attraction_Candidates,
        Attraction_Refinement_Step1,
        Attraction_Refinement_Step2,
        Attraction_Refinement_Step3,
        Attraction_Refinement_Step4,
        Attraction_Refinement_Step5,
        create_dynamic_select_tool
    )
    from env.core.data_types import TimeInfo

    def run_task_test(
        task_name,
        decide_tool,
        search_tool,
        refine_steps,
        decide_input,
        refinement_levels=(1, 3, 5)
    ):
        print(f"\n=== {task_name.upper()} TOOL FUNCTIONALITY TEST ===")
        # Decide
        print("\n[DECIDE TOOL]")
        decide_args = decide_tool.process_input(json.dumps(decide_input))
        pref_obj = decide_tool.execute(*decide_args)
        print("process_input:", decide_args)
        print("execute:", pref_obj)
        print("process_output:", decide_tool.process_output(pref_obj, refinement_level=0))

        # Search
        print("\n[SEARCH TOOL]")
        time_info_obj = TimeInfo.create("2025-09-01", "2025-09-10")
        search_input = {
            "preference_id": pref_obj.unique_id,
            "departure_time_id": time_info_obj.departure_time_id,
            "return_time_id": time_info_obj.return_time_id
        }
        search_args = search_tool.process_input(json.dumps(search_input))
        search_obj = search_tool.execute(*search_args)
        print("process_input:", search_args)
        print("execute:", search_obj)
        print("process_output:", search_tool.process_output(search_obj, refinement_level=0))

        # Refine chain & Select for each level
        print("\n[REFINE & SELECT TOOLS]")
        for level in refinement_levels:
            print(f"\n--- {task_name.capitalize()} Refine Chain (level {level}) ---")
            last_obj = search_obj
            for step in range(1, level + 1):
                refine_tool = refine_steps[step]
                # Check whether TimeInfo is required
                refine_input = {"preference_id": last_obj.unique_id}
                if "TimeInfo" in refine_tool.input_types:
                    refine_input["departure_time_id"] = time_info_obj.departure_time_id
                    refine_input["return_time_id"] = time_info_obj.return_time_id
                refine_args = refine_tool.process_input(json.dumps(refine_input))
                refine_obj = refine_tool.execute(*refine_args)
                print(f"Refine step {step} process_input:", refine_args)
                print(f"Refine step {step} execute:", refine_obj)
                print(f"Refine step {step} process_output:", refine_tool.process_output(refine_obj, refinement_level=step))
                last_obj = refine_obj

            # Select Tool
            select_tool = create_dynamic_select_tool(task_name.lower(), refinement_level=level)
            select_input = {"preference_id": last_obj.unique_id}
            select_args = select_tool.process_input(json.dumps(select_input))
            select_obj = select_tool.execute(*select_args)
            print("Select process_input:", select_args)
            print("Select execute:", select_obj)
            print("Select process_output:", select_tool.process_output(select_obj, refinement_level=level))


    # Location
    run_task_test(
    "location",
    Decide_Location_Preference,
    Search_Location_Candidates,
    refine_steps={
        1: Location_Refinement_Step1,
        2: Location_Refinement_Step2,
        3: Location_Refinement_Step3,
        4: Location_Refinement_Step4,
        5: Location_Refinement_Step5,
    },
    decide_input={
        "category": "city",
        "tier": "bustling_metropolis",
        "style": "adventure",
        "feature_package": "history_buff"
    }
)

    # Transportation
    run_task_test(
        "transportation",
        Decide_Transportation_Preference,
        Search_Transportation_Candidates,
        refine_steps={
            1: Transportation_Refinement_Step1,
            2: Transportation_Refinement_Step2,
            3: Transportation_Refinement_Step3,
            4: Transportation_Refinement_Step4,
            5: Transportation_Refinement_Step5,
        },
        decide_input={
            "category": "flight",
            "tier": "economy",
            "style": "speed_priority",
            "feature_package": "direct_route"
        }
    )
    
    # Test if the id system works well
    run_task_test(
        "transportation",
        Decide_Transportation_Preference,
        Search_Transportation_Candidates,
        refine_steps={
            1: Transportation_Refinement_Step1,
            2: Transportation_Refinement_Step2,
            3: Transportation_Refinement_Step3,
            4: Transportation_Refinement_Step4,
            5: Transportation_Refinement_Step5,
        },
        decide_input={
            "category": "flight",
            "tier": "economy",
            "style": "speed_priority",
            "feature_package": "direct_route"
        }
    )

    # Accommodation
    run_task_test(
        "accommodation",
        Decide_Accommodation_Preference,
        Search_Accommodation_Candidates,
        refine_steps={
            1: Accommodation_Refinement_Step1,
            2: Accommodation_Refinement_Step2,
            3: Accommodation_Refinement_Step3,
            4: Accommodation_Refinement_Step4,
            5: Accommodation_Refinement_Step5,
        },
        decide_input={
            "category": "hotel",
            "tier": "standard",
            "style": "city_center",
            "feature_package": "business_facilities"
        }
    )

    # Attraction
    run_task_test(
        "attraction",
        Decide_Attraction_Preference,
        Search_Attraction_Candidates,
        refine_steps={
            1: Attraction_Refinement_Step1,
            2: Attraction_Refinement_Step2,
            3: Attraction_Refinement_Step3,
            4: Attraction_Refinement_Step4,
            5: Attraction_Refinement_Step5,
        },
        decide_input={
            "category": "museum",
            "tier": "must_see",
            "style": "educational",
            "feature_package": "guided_tour"
        }
    )

    print("\nAll tool function tests finished.")