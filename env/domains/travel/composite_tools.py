"""
Composite tool definitions for the travel domain.

Defines composite tools used in travel planning. Each composite tool consists of multiple atomic tools executed sequentially to complete complex subtasks.

Composite tool characteristics:
1. Consists of 2-7 atomic tools
2. Executes sequentially, passing each tool's output to the next
3. Automatically validates type compatibility across the chain
4. Supports configurable refinement levels
"""

import sys
import os
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)

import numpy as np
from typing import List, Dict, Any, Optional, Union
from env.core.base_types import Tool, AtomicTool
from env.domains.travel.atomic_tools import get_all_subtasks
from env.domains.travel.refinement_steps_catalog import get_max_refinement_depth
import json

# =====================================================
# Global variables are managed dynamically
# =====================================================

# Composite tool registry is generated dynamically and initialized as empty
COMPOSITE_TOOLS_REGISTRY = {}



# =====================================================
# Composite tool class definition
# =====================================================

def parse_model_outputs_composite(model_outputs: str) -> Any:
        """
        Composite tools require no extra processing; return the input directly.
        
        Args:
            model_outputs: Model output string
            
        Returns:
            Any: Dictionary that satisfies the composite tool input requirements
                (
                    e.g. {
                        "LocationPreference": LocationPreference(...),
                        "TimeInfo": TimeInfo(...),
                        ...
                    }
                )
        """
        try:
            # First attempt to parse directly
            args = json.loads(model_outputs)
            # If the parsed result is still a string, decode repeatedly up to a limit
            depth = 0
            max_depth = 3
            while isinstance(args, str) and depth < max_depth:
                try:
                    args = json.loads(args)
                except Exception:
                    break
                depth += 1
        except Exception as e:
            # If direct parsing fails, attempt to sanitize the string
            try:
                # Trim potential leading/trailing whitespace or newlines
                cleaned_outputs = model_outputs.strip()
                
                # Handle single-layer quoted JSON strings (e.g. "{\"key\": \"value\"}" -> {"key": "value"})
                # This is the most common failure pattern
                if (cleaned_outputs.startswith('"') and cleaned_outputs.endswith('"') and 
                    cleaned_outputs.count('"') > 2):
                    # Remove the outer quotes
                    inner = cleaned_outputs[1:-1]
                    
                    # Check whether the inner content is valid JSON
                    if inner.startswith('{') and inner.endswith('}'):
                        # Handle escaped characters within the payload
                        cleaned_outputs = inner.replace('\\"', '"')
                    elif inner.startswith('\\"') and inner.endswith('\\"'):
                        # Handle the "\\"...\\"" pattern
                        cleaned_outputs = inner[2:-2].replace('\\\\"', '"').replace('\\\\', '\\')
                    else:
                        # If the inner content is not standard JSON, leave it for later handling
                        pass
                
                # Handle overly escaped JSON strings (e.g. "\\"{...}\\"" -> "{...}")
                elif cleaned_outputs.startswith('"\\"') and cleaned_outputs.endswith('\\""'):
                    # Remove the outer quotes and unescape
                    cleaned_outputs = cleaned_outputs[2:-2]  # Remove "\\" and \\""
                    # Replace \\\\" with " and \\\\ with \\
                    cleaned_outputs = cleaned_outputs.replace('\\\\"', '"').replace('\\\\', '\\')
                
                # If the string does not start with {, attempt to locate the JSON portion
                if not cleaned_outputs.startswith('{'):
                    # Look for the first { and the last }
                    start_idx = cleaned_outputs.find('{')
                    end_idx = cleaned_outputs.rfind('}')
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        cleaned_outputs = cleaned_outputs[start_idx:end_idx+1]
                    else:
                        # If a complete JSON structure is not found, attempt to construct one
                        # Handles cases like '\n    "tool_name": "test"\n'
                        if ':' in cleaned_outputs and '"' in cleaned_outputs:
                            # Try wrapping the snippet as a complete JSON object
                            cleaned_outputs = '{' + cleaned_outputs + '}'
                
                # Handle special trailing markers (e.g. <｜tool▁call▁end｜>)
                special_endings = ['<｜tool▁call▁end｜>', '<|tool_call_end|>', '<|end|>']
                for ending in special_endings:
                    if ending in cleaned_outputs:
                        cleaned_outputs = cleaned_outputs.split(ending)[0]
                
                # Handle JSON without outer quotes but containing escaped text (e.g. {\"key\": \"value\"})
                if cleaned_outputs.startswith('{') and '\\"' in cleaned_outputs and not cleaned_outputs.startswith('{"'):
                    # Replace the inner \\\\" with "
                    cleaned_outputs = cleaned_outputs.replace('\\"', '"')
                
                args = json.loads(cleaned_outputs)
                # If the parsed result is still a string, perform repeated decoding
                depth = 0
                max_depth = 3
                while isinstance(args, str) and depth < max_depth:
                    try:
                        args = json.loads(args)
                    except Exception:
                        break
                    depth += 1
            except Exception as e2:
                # Final attempt: handle potential string formatting issues
                try:
                    # Attempt to resolve possible Unicode escapes or other encoding problems
                    final_attempt = model_outputs.strip()
                    
                    # Handle potential multi-layer escaping
                    if final_attempt.count('\\') > final_attempt.count('"'):
                        # If the string appears over-escaped, unescape step by step
                        import re
                        # First handle \\" -> \"
                        final_attempt = re.sub(r'\\\\"', r'\"', final_attempt)
                        # Then handle \" -> "
                        final_attempt = re.sub(r'\\"', r'"', final_attempt)
                        # Handle \\\\ -> \\
                        final_attempt = re.sub(r'\\\\\\', r'\\', final_attempt)
                        # Handle \\ -> \
                        final_attempt = re.sub(r'\\\\', r'\\', final_attempt)
                    
                    # Remove potential leading and trailing quotes
                    if final_attempt.startswith('"') and final_attempt.endswith('"'):
                        final_attempt = final_attempt[1:-1]
                    
                    # Ensure the result resembles valid JSON
                    if not final_attempt.startswith('{'):
                        start_idx = final_attempt.find('{')
                        if start_idx != -1:
                            end_idx = final_attempt.rfind('}')
                            if end_idx > start_idx:
                                final_attempt = final_attempt[start_idx:end_idx+1]
                        else:
                            # If '{' is missing but ':' and '"' appear, attempt to repair partial JSON
                            if ':' in final_attempt and '"' in final_attempt:
                                # Check whether the leading {" is missing
                                if not final_attempt.startswith('"') and final_attempt.count('"') >= 2:
                                    final_attempt = '{"' + final_attempt
                                elif final_attempt.startswith('"') and not final_attempt.startswith('{"'):
                                    final_attempt = '{' + final_attempt
                    
                    args = json.loads(final_attempt)
                    # If the parsed result is still a string, perform repeated decoding
                    depth = 0
                    max_depth = 3
                    while isinstance(args, str) and depth < max_depth:
                        try:
                            args = json.loads(args)
                        except Exception:
                            break
                        depth += 1
                except Exception as e3:
                    print(f"[ERROR] Composite tool JSON parsing failed for model_outputs: {repr(model_outputs)}")
                    print(f"[ERROR] Original error: {e}")
                    print(f"[ERROR] Cleaned parsing error: {e2}")
                    print(f"[ERROR] Final attempt error: {e3}")
                    from env.core.base_types import JSON_DECODE_ERROR_OUTPUT
                    return JSON_DECODE_ERROR_OUTPUT

        # Ensure args is a dictionary
        if not isinstance(args, dict):
            print(f"[ERROR] Composite tool parsed result is not a dictionary: {type(args)} - {repr(args)}")
            from env.core.base_types import JSON_DECODE_ERROR_OUTPUT
            return JSON_DECODE_ERROR_OUTPUT
        
        return {k: str(v) for k, v in args.items()}

class CompositeTool(Tool):
    def __init__(
        self, 
        component_tool_names: List[str],
        refinement_level: int = None,
        cost: Optional[int] = None
    ):
        """
        Initialize a composite tool.
        
        Args:
            component_tool_names: List of component atomic tool names.
            refinement_level: Refinement level used to obtain the dynamic tool registry. Defaults to the maximum depth when None.
            cost: Tool cost; automatically calculated when None.
        """
        # Must contain at least two atomic tools
        if len(component_tool_names) < 2:
            raise ValueError("[CODE ERROR] Composite tool must contain at least one atomic tool")
        
        # 1. Set core attributes first (refinement_level must be set before validation)
        self.component_tool_names = component_tool_names
        self.component_count = len(component_tool_names)
        # Use the maximum refinement depth if not specified
        self.refinement_level = refinement_level if refinement_level is not None else get_max_refinement_depth()
        
        # 2. Validate the existence and compatibility of atomic tools
        # Ignore the available_tools argument here; pull directly from atomic tools
        from env.domains.travel.atomic_tools import get_all_dynamic_atomic_tools
        all_tools_dict = get_all_dynamic_atomic_tools(self.refinement_level)
        if not self._validate_component_tools(available_tools=all_tools_dict):
            raise ValueError("[CODE ERROR] One or more atomic tools are invalid or incompatible")
        
        # 3. Derive all other attributes
        name = self._generate_name()
        input_types = self._infer_input_types()
        output_type = self._infer_output_type()
        description = self._generate_description()
        
        # 5. Call the parent constructor
        if cost is None:
            super().__init__(name, input_types, output_type, None, description)
        else:
            super().__init__(name, input_types, output_type, cost, description)

    def _validate_component_tools(self, available_tools: Dict[str, Tool]) -> bool:
        """
        Validate the effectiveness of the component tools.
        
        Validation rules:
        1. Ensure each atomic tool exists.
        2. Ensure every output from an atomic tool is consumed by the next tool in the chain.
        
        Returns:
            bool: True if all component tools are valid and compatible.
        """
        try:
            
            all_tools = available_tools
            
            # 1. Ensure each tool exists
            for tool_name in self.component_tool_names:
                if tool_name not in all_tools:
                    print(f"[CODE ERROR] Unknown atomic tool: {tool_name}")
                    return False
            
            # 2. If there is only one tool, existence implies validity
            if len(self.component_tool_names) < 2:
                return True
            
            # 3. Retrieve the atomic tool objects
            atomic_tools = [available_tools[name] for name in self.component_tool_names]
            
            # 4. Validate chain compatibility: each tool's output must be consumed by the next tool
            for i in range(len(atomic_tools) - 1):
                current_tool = atomic_tools[i]
                next_tool = atomic_tools[i + 1]
                current_output = current_tool.output_type
                next_inputs = next_tool.input_types
                
                # Check whether the current tool's output appears in the next tool's inputs
                if current_output not in next_inputs:
                    print(f"[CODE ERROR] {current_tool.name} output '{current_output}' not found in {next_tool.name} inputs {next_inputs}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"[CODE ERROR] Tool lookup error during validation: {e}")
            return False
    
    def _generate_name(self) -> str:
        """
        Generate the composite tool name.
        
        Returns:
            str: The generated name.
        """
        if len(self.component_tool_names) == 1:
            return self.component_tool_names[0]
        
        subtask = self._infer_subtask()
        subtask_cap = subtask.capitalize()
        tool_names = self.component_tool_names
        
        # Reuse the original generate_composite_tool_name logic
        if len(tool_names) == 2:
            first_tool = tool_names[0]
            second_tool = tool_names[1]
            
            if first_tool.startswith(f"Decide_{subtask_cap}_") and second_tool.startswith(f"Search_{subtask_cap}_"):
                return f"{subtask_cap}_Preference_and_Search"
            elif second_tool.startswith(f"{subtask_cap}_Refinement_"):
                step_match = second_tool.split("_")[-1]
                if first_tool.startswith(f"Decide_{subtask_cap}_"):
                    return f"{subtask_cap}_Planning_to_{step_match}"
                elif first_tool.startswith(f"Search_{subtask_cap}_"):
                    return f"{subtask_cap}_Search_to_{step_match}"
                elif first_tool.startswith(f"{subtask_cap}_Refinement_"):
                    return f"{subtask_cap}_Refine_to_{step_match}"
                else:
                    return f"{subtask_cap}_Planning_to_{step_match}"
            elif second_tool.startswith("Select_Final_"):
                # Two-step chain + Select: if the first step is Search, use the clearer from_Search label; otherwise keep the original rule
                if first_tool.startswith(f"Search_{subtask_cap}_"):
                    return f"{subtask_cap}_Finalize_from_Search"
                elif "_Step" in first_tool:
                    step_match = first_tool.split("_")[-1]
                    return f"{subtask_cap}_Finalize_from_{step_match}"
                else:
                    return f"{subtask_cap}_Finalize_from_Raw"
            else:
                start_idx = 0
                if first_tool.startswith(f"Decide_{subtask_cap}_"):
                    start_idx = 1
                elif first_tool.startswith(f"Search_{subtask_cap}_"):
                    start_idx = 2
                elif "_Refinement_Step" in first_tool:
                    start_idx = int(first_tool.split("_")[-1][-1]) + 2
                
                return f"{subtask_cap}_Composite_2Tools_From_{start_idx}"
        
        else:
            # Combination of multiple tools
            last_tool = tool_names[-1]
            first_tool = tool_names[0]
            
            if last_tool.startswith(f"{subtask_cap}_Refinement_"):
                step_match = last_tool.split("_")[-1]
                if first_tool.startswith(f"Decide_{subtask_cap}_"):
                    return f"{subtask_cap}_Full_Planning_to_{step_match}"
                elif first_tool.startswith(f"Search_{subtask_cap}_"):
                    return f"{subtask_cap}_Search_Planning_to_{step_match}"
                else:
                    start_step = first_tool.split("_")[-1] if "_Step" in first_tool else "Raw"
                    return f"{subtask_cap}_Refine_{start_step}_to_{step_match}"
                     
            elif last_tool.startswith("Select_Final_"):
                if first_tool.startswith(f"Decide_{subtask_cap}_"):
                    return f"{subtask_cap}_Complete_{len(tool_names)}Steps_Pipeline"
                elif first_tool.startswith(f"Search_{subtask_cap}_"):
                    return f"{subtask_cap}_Search_to_Final_{len(tool_names)}Steps"
                else:
                    start_step = first_tool.split("_")[-1] if "_Step" in first_tool else "Raw"
                    return f"{subtask_cap}_Finish_from_{start_step}_{len(tool_names)}Steps"
            else:
                start_info = "Decide" if first_tool.startswith(f"Decide_{subtask_cap}_") else \
                            "Search" if first_tool.startswith(f"Search_{subtask_cap}_") else \
                            first_tool.split("_")[-1] if "_Step" in first_tool else "Unknown"
                return f"{subtask_cap}_Chain_{len(tool_names)}Tools_From_{start_info}"
    
    def _generate_description(self) -> str:
        """
        Generate the tool description.
        
        Returns:
            str: The generated description.
        """
        from env.domains.travel.atomic_tools import get_all_dynamic_atomic_tools
        from env.domains.travel.composite_tool_descriptions import COMPOSITE_DESCRIPTIONS
        
        if len(self.component_tool_names) == 1:
            raise ValueError("[CODE ERROR] Composite tool must contain at least two atomic tools")
        
        # 1. First try to fetch a predefined description
        tool_names_tuple = tuple(self.component_tool_names)
        if tool_names_tuple in COMPOSITE_DESCRIPTIONS:
            return COMPOSITE_DESCRIPTIONS[tool_names_tuple]
        
        # 2. If not available, generate the description dynamically
        print(f"[WARNING] No predefined description found for {tool_names_tuple}, generating dynamically...")
        
        subtask = self._infer_subtask()
        dynamic_tools = get_all_dynamic_atomic_tools(self.refinement_level)
        tool_descriptions = [dynamic_tools[name].description for name in self.component_tool_names]
        
        # Produce the description
        if len(self.component_tool_names) == 2:
            return f"Execute the compound operation of {subtask} subtask: {tool_descriptions[0]}, then {tool_descriptions[1]}"
        elif len(self.component_tool_names) <= 4:
            steps_desc = ", ".join([f"step {i+1}: {desc}" for i, desc in enumerate(tool_descriptions)])
            return f"Execute the compound operation of {subtask} subtask ({len(self.component_tool_names)} steps in total): {steps_desc}"
        else:
            # Handle lengthy descriptions
            first_desc = tool_descriptions[0]
            last_desc = tool_descriptions[-1]
            middle_keywords = []
            for desc in tool_descriptions[1:-1]:
                if "Refinement" in desc:
                    middle_keywords.append("Refinement")
                elif "Search" in desc:
                    middle_keywords.append("Search")
                elif "Validation" in desc:
                    middle_keywords.append("Validation")
                elif "Evaluation" in desc:
                    middle_keywords.append("Evaluation")
                else:
                    middle_keywords.append("Processing")
            
            unique_keywords = list(dict.fromkeys(middle_keywords))
            middle_desc = ", ".join(unique_keywords) if unique_keywords else "Processing"
            
            return (f"[INFO] Execute the compound operation of {subtask} subtask (total {len(self.component_tool_names)} steps): {first_desc}, "
                    f"followed by {middle_desc} steps, and finally {last_desc}")
    
    def _infer_input_types(self) -> List[str]:
        """
        Infer the input types required by the composite tool.
        
        Returns:
            List[str]: All input types required by the composite tool.
            
        Logic: Remove input types already provided by previous tool outputs; the remaining types are required inputs for the composite tool.
        """
        from env.domains.travel.atomic_tools import get_all_dynamic_atomic_tools
        
        dynamic_tools = get_all_dynamic_atomic_tools(self.refinement_level)
        atomic_tools = [dynamic_tools[name] for name in self.component_tool_names]
        
        required_inputs = set()
        available_outputs = set()
        for tool in atomic_tools:
            for input_type in tool.input_types:
                if input_type not in available_outputs:
                    required_inputs.add(input_type)
            available_outputs.add(tool.output_type)
        return sorted(list(required_inputs))
    
    def _infer_output_type(self) -> str:
        """
        Infer the composite tool's output type (the final tool's output type).
        
        Returns:
            str: The composite tool output type.
        """
        from env.domains.travel.atomic_tools import get_all_dynamic_atomic_tools
        
        dynamic_tools = get_all_dynamic_atomic_tools(self.refinement_level)
        last_tool = dynamic_tools[self.component_tool_names[-1]]
        return last_tool.output_type
    
    def _infer_subtask(self) -> str:
        """Infer the subtask type from the tool names."""
        first_tool = self.component_tool_names[0]
        if "Location" in first_tool:
            return "location"
        elif "Transportation" in first_tool:
            return "transportation"
        elif "Accommodation" in first_tool:
            return "accommodation"
        elif "Attraction" in first_tool:
            return "attraction"
        elif "Dining" in first_tool:
            return "dining"
        elif "Shopping" in first_tool:
            return "shopping"
        else:
            raise ValueError(f"[CODE ERROR] Cannot infer subtask from {first_tool}")
    
    def _calculate_cost(self) -> Optional[float]:
        """
        Calculate the cost of the composite tool.
        
        Returns:
            Optional[float]: The calculated cost, or None if any component tool lacks a defined cost.
        """
        from env.domains.travel.atomic_tools import get_all_dynamic_atomic_tools
        
        total = 0
        k = len(self.component_tool_names)
        
        # Retrieve the dynamic tool registry
        dynamic_tools = get_all_dynamic_atomic_tools(self.refinement_level)
        
        for tool_name in self.component_tool_names:
            try:
                if tool_name in dynamic_tools:
                    tool = dynamic_tools[tool_name]
                else:
                    print(f"[CODE ERROR] CompositeTool._calculate_cost(): Tool {tool_name} not found in dynamic registry")
                    return None
                
                if not tool.has_cost():
                    print(f"[CODE ERROR] CompositeTool._calculate_cost(): Tool {tool_name} has no cost defined")
                    return None
                total += tool.cost
                    
            except Exception as e:
                print(f"[CODE ERROR] Error getting cost for tool {tool_name}: {e}")
                return None
        
        # Add Gaussian noise
        from env.settings import load_config
        import math

        noise_std = load_config().tool_defaults.noise_std * math.sqrt(k)
        import hashlib
        import numpy as np
        seed_val = int(hashlib.sha256(f"composite_runtime_noise:{self.name}:{self.refinement_level}".encode("utf-8")).hexdigest(), 16) % (2**31 - 1)
        rng = np.random.default_rng(seed_val)
        noise = rng.normal(0, noise_std)
        final_cost = max(1.00, round(total + noise, 2))
        
        return final_cost
    
    # def get_atomic_tools(self, refinement_level: int = None) -> List[AtomicTool]:
    #     """
    #     Retrieve the list of component atomic tool objects.
    #     
    #     Args:
    #         refinement_level: Refinement level, defaults to the initialization level when None.
    #     
    #     Returns:
    #         List[AtomicTool]: List of atomic tool objects.
    #     """
    #     from env.domains.travel.atomic_tools import get_all_dynamic_atomic_tools
        
    #     if refinement_level is None:
    #         refinement_level = getattr(self, 'refinement_level', 5)
        
    #     # Use the dynamic tool registry
    #     dynamic_tools = get_all_dynamic_atomic_tools(refinement_level)
    #     return [dynamic_tools[name] for name in self.component_tool_names]
        
    def process_input(self, model_outputs: str) -> Dict[str, Any]:
        """
        Process inputs (composite tools require no extra handling; return the input directly).
        
        Args:
            model_outputs: Model output string
            
        Returns:
            Dict[str, Any]: Dictionary matching the composite tool's input requirements
                (
                    e.g. {
                        "LocationPreference": LocationPreference(...),
                        "TimeInfo": TimeInfo(...),
                        ...
                    }
                )
        """
        # First invoke the parser to obtain the identifier dictionary
        model_outputs = parse_model_outputs_composite(model_outputs)
        from env.core.base_types import JSON_DECODE_ERROR_OUTPUT
        if isinstance(model_outputs, str) and model_outputs == JSON_DECODE_ERROR_OUTPUT:  
            return {"error_message": model_outputs}  
        if model_outputs is None:
            print("[RUNTIME ERROR] CompositeTool.process_input(): parse_model_outputs_composite returned None")
            return {"error_message": "[ERROR] composite_tools.py - parse_model_outputs_composite - Failed to parse model outputs."}
        
        if not isinstance(model_outputs, dict):
            print("[RUNTIME ERROR] Composite tool input must be a dictionary of input types")
            return {"error_message": "[ERROR] composite_tools.py - process_input - Invalid input format."}
 
        from env.core.data_types import DATA_TYPE_REGISTRY
         
        result = {}
         
        # For each composite tool input_type, try to fetch it from model_outputs
        for input_type in self.input_types:
            if input_type in model_outputs:
                # @identifier: string id/enum value
                identifier = model_outputs[input_type]
                 
                from env.domains.travel.enums import ENUM_DIRECT_MAPPINGS
                 
                # Check whether the type is known
                if input_type not in DATA_TYPE_REGISTRY and input_type not in ENUM_DIRECT_MAPPINGS:
                    print(f"[ERROR] composite_tools.py - CompositeTool.process_input(model_outputs) - Unknown input type '{input_type}'.")
                    return {"error_message": f"[ERROR] Input {input_type} is invalid."}
                     
                # Retrieve the data type class
                TypeClass = DATA_TYPE_REGISTRY.get(input_type) if input_type in DATA_TYPE_REGISTRY else ENUM_DIRECT_MAPPINGS[input_type]
                 
                # Create placeholder objects based on the type
                if input_type.endswith("Preference"): # Preference inputs imply search tools
                    # Preference types: create placeholders with IDs
                    obj = TypeClass(
                        unique_id=identifier,
                        category="",
                        tier="",
                        style="",
                        feature_package=""
                    )
                elif input_type == "TimeInfo":
                    obj = TypeClass(
                        unique_id=identifier,
                        departure_time_id="",
                        return_time_id=""
                    )
                elif "Candidate" in input_type:
                    # Candidate object types
                    if input_type.endswith("Candidate_Raw"):
                        obj = TypeClass(
                            unique_id=identifier,
                            preference_id=identifier,
                            preference=None
                        )
                    else:
                        obj = TypeClass(
                            unique_id=identifier,
                            previous_candidate_id=identifier,
                            previous_candidate=None
                        )
                elif input_type.endswith("Category") or input_type.endswith("Tier") or input_type.endswith("Style") or input_type.endswith("FeaturePackage"):
                    # Enum types
                    if identifier not in [member.value for member in TypeClass]:
                        print(f"[RUNTIME ERROR] composite_tools.py - CompositeTool.process_input(model_outputs): Invalid enum value '{identifier}' for type '{input_type}'")
                        return {"error_message": f"[ERROR] Invalid enum value '{identifier}' for type '{input_type}'."}
                    else:
                        obj = TypeClass(identifier)
                else:
                    print(f"[RUNTIME ERROR] CompositeTool.process_input(): Unsupported input type '{input_type}'")
                    return {"error_message": f"[ERROR] Input type '{input_type}' is invalid."}
                if obj:
                    result[input_type] = obj
                else:
                    print(f"[CODE/RUNTIME ERROR] CompositeTool.process_input(): Failed to create object for input type '{input_type}'")
                    return {"error_message": f"[ERROR] Failed to create object for input type '{input_type}'."}
            else:
                print(f"[RUNTIME ERROR] CompositeTool.process_input(), line 499: Required input type '{input_type}' not found in model outputs")
                return {"error_message": f"[ERROR] Required input type '{input_type}' not found in model's tool_calling outputs."}
 
        return result
    
    
    def execute(self, input_data: Dict[str, Any], available_tools: Dict[str, Tool]) -> Any:
        """
        Execute the composite tool by running each component tool sequentially.

        Args:
            input_data: Dictionary mapping type names to objects (e.g. {"TypeName": TypeObject, ...}).
            available_tools: Registry mapping tool names to tool objects.
            
        Returns:
            Any: Output produced by the final tool in the chain.
        """        
        # Validate tool availability
        from env.domains.travel.tool_registry import get_all_tools
        all_tools = get_all_tools(self.refinement_level)
        
        if not self._validate_component_tools(available_tools=all_tools):
            return "[ERROR] CompositeTool.execute(): One or more atomic tools are invalid or incompatible."
        if not self._validate_component_tools(available_tools=available_tools):
            return "[ERROR] CompositeTool.execute(): One or more atomic tools are not accessible."
        if "error_message" in input_data:
            return input_data["error_message"]
        
        # Maintain a cumulative dictionary of available inputs, including previous outputs
        available_inputs = input_data.copy()
        current_output = None
        
        # Execute each component tool sequentially
        for i, tool_name in enumerate(self.component_tool_names):
            
            if tool_name not in available_tools:
                return f"[ERROR] Tool {tool_name} not found in the current 'available tools'."
 
            # Look up the current tool
            tool = available_tools[tool_name]
            # Initialize the current tool's input dictionary
            tool_input = {}
             
            # Build inputs for the current tool
            for input_type in tool.input_types:
                if input_type in available_inputs:
                    # Pull from the cumulative inputs, including prior outputs
                    tool_input[input_type] = available_inputs[input_type]
                else:
                    print(f"[ERROR] CompositeTool.execute(), line 547: Required input type '{input_type}' not found for tool '{tool_name}'. Available inputs: {list(available_inputs.keys())}")
                    return f"[ERROR] Required input type '{input_type}' not found for tool '{tool_name}'. Available inputs: {list(available_inputs.keys())}"
             
            # Execute the current tool
            current_output = tool.execute(input_data=tool_input, available_tools=available_tools)
            # Abort immediately if the current step returns an error message
            if isinstance(current_output, str) and current_output.startswith("[ERROR]"):
                print(f"[ERROR] CompositeTool.execute(), line 554: Something went wrong in the tool chain of {self.name}, and the atomic tool's error message is {current_output}")
                return f"[ERROR] Invalid input for {self.name}"
             
            # Add the current tool's output to the pool of available inputs
            if current_output is not None:
                output_type = type(current_output).__name__
                available_inputs[output_type] = current_output
 
        return current_output
    

    def process_output(self, tool_output: Any, refinement_level: int, available_tools: Dict[str, Tool]) -> str:
        """
        Process the output from the composite tool.
        Generates a two-part narrative:
        1. Summarizes which intermediate steps were executed (e.g. search -> filter).
        2. Delegates to the final atomic tool's process_output for a detailed result description.
        
        Args:
            tool_output: Output returned by the final atomic tool.
            refinement_level: Refinement level (0 for initial output, 1 for first refinement, etc.); kept for API consistency.
            available_tools: Registry mapping tool names to tool objects.
            
        Returns:
            str: Formatted description string.
        """
        # Basic error handling: if the input is already an error message, return it
        if isinstance(tool_output, str) and tool_output.startswith("[ERROR]"):
            return tool_output
         
        # Guard against an empty component tool list
        if not self.component_tool_names:
            # return f"[ERROR] This tool has no component tools."
            raise ValueError("[CODE ERROR] This tool has no component tools.")
 
        # --- Step 1: build the detailed description using the final atomic tool ---
        # Retrieve the final atomic tool to invoke its process_output implementation
        from env.domains.travel.atomic_tools import get_all_dynamic_atomic_tools
        all_tools = get_all_dynamic_atomic_tools(self.refinement_level)
        last_tool_name = self.component_tool_names[-1]
        
        if last_tool_name not in all_tools:
            return f"[ERROR] Could not find the last tool '{last_tool_name}' in the all atomic tools."
        if last_tool_name not in available_tools:
            return f"[ERROR] The last tool '{last_tool_name}' is not accessible now."
            
        last_tool = available_tools[last_tool_name]
        
        # Invoke the final atomic tool's process_output using its expected signature
        final_output_description = last_tool.process_output(
            tool_output=tool_output, 
            refinement_level=self.refinement_level,
            available_tools=available_tools
        )
 
        # --- Step 2: summarize the intermediate steps ---
        intermediate_steps = []
        # Traverse every component tool except the last
        for tool_name in self.component_tool_names[:-1]:
            # Generate concise action descriptions based on tool names
            if "Decide" in tool_name:
                intermediate_steps.append("Determined Preferences")
            elif "Search" in tool_name:
                intermediate_steps.append("Initial Search")
            elif "Refinement" in tool_name:
                # Attempt to infer the refinement dimension similarly to atomic tools
                tool = available_tools.get(tool_name)
                if tool:
                    try:
                        # Assume the filter dimension is encoded in the output_type
                        filter_dimension_map = {
                            1: "availability", 
                            2: "location", 
                            3: "reputation", 
                            4: "restrictions", 
                            5: "cost-effectiveness"
                        } 
                        from env.core.base_types import get_all_digits
                        step_num = int(get_all_digits(tool.output_type))
                        dimension = filter_dimension_map.get(step_num, "filter")
                        intermediate_steps.append(f"Filtered by {dimension}")
                    except:
                        intermediate_steps.append("Refined results")  # Fallback on failure
                else:
                    intermediate_steps.append("Refined results")
            # Additional tool categories can be supported with more elif branches
         
        # If the final tool is a refinement step:
        # 1) Include its filter dimension in the process summary;
        # 2) Remove any duplicated "Filtered by ..." prefix from the final description.
        if "Refinement" in last_tool_name:
            try:
                last_tool_obj = available_tools.get(last_tool_name)
                if last_tool_obj:
                    filter_dimension_map = {
                        1: "availability",
                        2: "location",
                        3: "reputation",
                        4: "restrictions",
                        5: "cost-effectiveness"
                    }
                    from env.core.base_types import get_all_digits
                    last_step_num = int(get_all_digits(last_tool_obj.output_type))
                    last_dimension = filter_dimension_map.get(last_step_num, "filter")
                    intermediate_steps.append(f"Filtered by {last_dimension}")
                # Strip duplicated "Filtered by ..." prefixes from the final description
                if isinstance(final_output_description, str) and final_output_description.startswith("Filtered by "):
                    dot_idx = final_output_description.find(".")
                    if dot_idx != -1:
                        final_output_description = final_output_description[dot_idx+1:].lstrip()
            except Exception:
                pass
        elif "Select" in last_tool_name:
            # Include the selection action in the process summary
            intermediate_steps.append("Final Selection")
 
        # --- Step 3: assemble the narrative ---
        if not intermediate_steps:
            raise ValueError("[CODE ERROR] CompositeTool.process_output(): No intermediate steps found, at least one expected.")
        else:
            process_summary = " -> ".join(intermediate_steps)
            return f"Process: {process_summary}. {final_output_description}"

    
    def get_tool_definition(self) -> Dict[str, Any]:
        """
        Retrieve the tool definition.
        
        Returns:
            Dict[str, Any]: Dictionary describing the tool.
        """
        return {
            "name": self.name,
            "type": "composite",
            "component_tool_names": self.component_tool_names,  # Matches the structure in to_dict()
            "component_count": self.component_count,
            "input_types": self.input_types,
            "output_type": self.output_type,
            "cost": self.cost,
            "description": self.description,
            "refinement_level": getattr(self, 'refinement_level', get_max_refinement_depth())
        }
    
    def get_tool_type(self) -> str:
        """Return the tool type."""
        return "composite"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool to a dictionary representation."""
        return {
            "name": self.name,
            "type": "composite",
            "component_tool_names": self.component_tool_names,
            "component_count": self.component_count,
            "input_types": self.input_types,
            "output_type": self.output_type,
            "cost": self.cost,
            "description": self.description,
            "refinement_level": getattr(self, 'refinement_level', get_max_refinement_depth()),
            # Optional: include additional metadata to aid understanding
            "is_location_dependent": any(name.startswith(("Transportation_", "Accommodation_", "Attraction_", "Dining_", "Shopping_")) 
                                    for name in self.component_tool_names),
            "contains_search_tools": any("Search_" in name for name in self.component_tool_names)
        }
    
    # def get_validation_errors(self) -> List[str]:
    #     """Return validation error details."""
    #     errors = []
    #     try:
    #         atomic_tools = self.get_atomic_tools()

    #         if len(atomic_tools) < 2:
    #             return errors  # Single-tool chains are considered compatible

    #         # Verify composite input_types match the union of atomic tool inputs
    #         all_atomic_inputs = set()
    #         for tool in atomic_tools:
    #             all_atomic_inputs.update(tool.input_types)

    #         expected_composite_inputs = sorted(list(all_atomic_inputs))
    #         if sorted(self.input_types) != expected_composite_inputs:
    #             errors.append(f"[CODE ERROR] CompositeTool {self.name} input_types mismatch. Expected: {expected_composite_inputs}, Actual: {sorted(self.input_types)}")

    #         # Validate type compatibility across the chain
    #         for i in range(len(atomic_tools) - 1):
    #             current_tool = atomic_tools[i]
    #             next_tool = atomic_tools[i + 1]
    #             current_output = current_tool.output_type
    #             next_inputs = next_tool.input_types

    #             if i == 0:
    #                 # First tool: its inputs must appear in the composite input list
    #                 for input_type in current_tool.input_types:
    #                     if input_type not in self.input_types:
    #                         errors.append(f"[CODE ERROR] First tool {current_tool.name} requires {input_type} not in composite inputs {self.input_types}")

    #             # Check whether the current tool output satisfies the next tool's input requirements
    #             if next_tool.name == "Search_Location_Candidates":
    #                 # Needs only the previous output and TimeInfo
    #                 if current_output not in next_inputs:
    #                     errors.append(f"[CODE ERROR] Search_Location_Candidates missing required input {current_output}")
    #                 if "TimeInfo" not in next_inputs:
    #                     errors.append(f"[CODE ERROR] Search_Location_Candidates missing TimeInfo input")
    #             elif next_tool.name.startswith("Search_"):
    #                 # Other search tools require LocationPreference and TimeInfo
    #                 if current_output not in next_inputs:
    #                     errors.append(f"[CODE ERROR] Search tool {next_tool.name} missing required input {current_output}")
    #                 if "TimeInfo" not in next_inputs:
    #                     errors.append(f"[CODE ERROR] Search tool {next_tool.name} missing TimeInfo input")
    #                 if "LocationPreference" not in next_inputs:
    #                     errors.append(f"[CODE ERROR] Search tool {next_tool.name} missing LocationPreference input")
    #             else:
    #                 # General case: previous outputs should appear among the next tool's inputs
    #                 if current_output not in next_inputs:
    #                     errors.append(f"[CODE ERROR] Type mismatch: {current_tool.name} outputs {current_output}, but {next_tool.name} expects {next_inputs}")

    #     except Exception as e:
    #         errors.append(f"[CODE ERROR] Tool lookup error: {e}")

    #     return errors

# =====================================================
# Composite tool generation functions
# =====================================================

def generate_continuous_subsequences(items: List[str], min_length: int = 2, max_length: int = None) -> List[List[str]]:
    """
    Generate every contiguous subsequence of a list.
    
    Args:
        items: Original list of items.
        min_length: Minimum subsequence length to include.
        max_length: Maximum subsequence length; includes all lengths when None.
        
    Returns:
        List[List[str]]: List of contiguous subsequences.
    """
    subsequences = []
    n = len(items)
    
    # Default to all possible lengths when max_length is not specified
    actual_max_length = max_length if max_length is not None else n
    
    for length in range(min_length, min(actual_max_length + 1, n + 1)):
        for start in range(n - length + 1):
            subsequence = items[start:start + length]
            subsequences.append(subsequence)
    
    return subsequences


def create_composite_tool_from_sequence(tool_names: List[str], refinement_level: int = None) -> CompositeTool:
    """
    Create a composite tool from an ordered sequence of tool names.
    
    Args:
        tool_names: List of tool names composing the chain.
        refinement_level: Refinement level at which to build the tool.
        
    Returns:
        CompositeTool: Newly constructed composite tool.
    """
    # The subtask argument is omitted because it is inferred automatically
    return CompositeTool(
        component_tool_names=tool_names,
        refinement_level=refinement_level
    )


def generate_subtask_composite_tools(subtask: str, refinement_level: int = None) -> Dict[str, CompositeTool]:
    """
    Generate all possible composite tools for a given subtask and refinement level.
    
    Args:
        subtask: Subtask name (e.g. "location").
        refinement_level: Refinement level (0-5).
        
    Returns:
        Dict[str, CompositeTool]: Dictionary of composite tools.
    """
    from env.domains.travel.atomic_tools import get_tools_by_subtask
    
    # Retrieve the tools for the subtask at this refinement level
    subtask_tools = get_tools_by_subtask(subtask, refinement_level)
    tool_chain_names = [tool.name for tool in subtask_tools]
    
    # Generate every contiguous subsequence
    subsequences = generate_continuous_subsequences(tool_chain_names, min_length=2)
    
    composite_tools = {}
    
    for subsequence in subsequences:
        try:
            # Create the composite tool (validation occurs in the constructor)
            composite_tool = CompositeTool(
                component_tool_names=subsequence,
                refinement_level=refinement_level
            )
            composite_tools[composite_tool.name] = composite_tool
                
        except Exception as e:
            print(f"[ERROR] Error creating composite tool from {subsequence}: {e}")
    
    return composite_tools


def generate_all_composite_tools(refinement_level: int = None) -> Dict[str, CompositeTool]:
    """
    Generate every composite tool (internal helper, not recommended for direct use).
    
    Args:
        refinement_level: Refinement level; defaults to the maximum depth when None.
        
    Returns:
        Dict[str, CompositeTool]: Dictionary of all composite tools.
    """
    
    # Default to the maximum refinement depth when not specified
    if refinement_level is None:
        refinement_level = get_max_refinement_depth()
    
    # print(f"Generating composite tools for refinement level L{refinement_level}...")
    
    all_tools = {}
    
    # Generate composite tools for each individual subtask
    # print("  - Generating single-subtask composite tools...")
    for subtask in get_all_subtasks():
        # print(f"    - {subtask} (L{refinement_level})...")
        subtask_tools = generate_subtask_composite_tools(subtask, refinement_level)
        all_tools.update(subtask_tools)
        # print(f"      Generated {len(subtask_tools)} tools")
    
    return all_tools


def get_all_dynamic_composite_tools(refinement_level: int = None) -> Dict[str, CompositeTool]:
    """
    Retrieve all dynamic composite tools for the given refinement level (primary public API).
    
    Args:
        refinement_level: Refinement level; defaults to the maximum depth when None.
        
    Returns:
        Dict[str, CompositeTool]: Dictionary containing all dynamic composite tools.
    """
    global COMPOSITE_TOOLS_REGISTRY
    
    # Default to the maximum refinement depth when not specified
    if refinement_level is None:
        refinement_level = get_max_refinement_depth()
    
    # print(f"[INFO] Generating dynamic composite tools for refinement level L{refinement_level}...")
    
    # Generate and return the composite tools for this level
    COMPOSITE_TOOLS_REGISTRY = generate_all_composite_tools(refinement_level)
    
    # print(f"[SUCCESS] Generated {len(COMPOSITE_TOOLS_REGISTRY)} composite tools for L{refinement_level}")
    
    return COMPOSITE_TOOLS_REGISTRY


# =====================================================
# Convenience lookup and management functions
# =====================================================

def get_composite_tool(tool_name: str, refinement_level: int = None) -> CompositeTool:
    """
    Retrieve a specific composite tool.
    
    Args:
        tool_name: Tool name.
        refinement_level: Refinement level.
        
    Returns:
        CompositeTool: The requested composite tool.
    """
    all_tools = get_all_dynamic_composite_tools(refinement_level)
    
    if tool_name not in all_tools:
        raise ValueError(f"[RUNTIME ERROR] Composite tool '{tool_name}' not found")
    
    return all_tools[tool_name]


def get_composite_tools_for_subtask(subtask: str, refinement_level: int = None) -> List[CompositeTool]:
    """
    Retrieve all composite tools for the specified subtask.
    
    Args:
        subtask: Subtask name.
        refinement_level: Refinement level.
        
    Returns:
        List[CompositeTool]: List of composite tools for the subtask.
    """
    all_tools = get_all_dynamic_composite_tools(refinement_level)
    subtask_cap = subtask.capitalize()
    
    return [tool for tool in all_tools.values() 
            if tool.name.startswith(subtask_cap)]


def get_composite_tools_by_complexity(complexity: int, refinement_level: int = None) -> List[CompositeTool]:
    """
    Retrieve all composite tools with the specified complexity (number of atomic components).
    
    Args:
        complexity: Number of atomic tools in the chain.
        refinement_level: Refinement level.
        
    Returns:
        List[CompositeTool]: List of composite tools matching the complexity.
    """
    all_tools = get_all_dynamic_composite_tools(refinement_level)
    
    return [tool for tool in all_tools.values() 
            if tool.component_count == complexity]


def get_tool_generation_statistics(refinement_level: int = None) -> Dict[str, Any]:
    """
    Gather statistics about generated composite tools.
    
    Args:
        refinement_level: Refinement level; defaults to the maximum depth when None.
        
    Returns:
        Dict[str, Any]: Statistics dictionary.
    """
    from env.domains.travel.atomic_tools import get_all_subtasks
    
    # Default to the maximum refinement depth when not provided
    if refinement_level is None:
        refinement_level = get_max_refinement_depth()
    
    all_tools = get_all_dynamic_composite_tools(refinement_level)
    
    if not all_tools:
        return {
            "total_composite_tools": 0,
            "avg_atomic_tools": 0,
            "min_atomic_tools": 0,
            "max_atomic_tools": 0,
            "subtask_tool_counts": {},
            "complexity_distribution": {}
        }
    
    # Compute basic statistics
    component_counts = [tool.component_count for tool in all_tools.values()]
    
    stats = {
        "total_composite_tools": len(all_tools),
        "avg_atomic_tools": sum(component_counts) / len(component_counts),
        "min_atomic_tools": min(component_counts),
        "max_atomic_tools": max(component_counts),
    }
    
    # Count by subtask
    subtask_counts = {}
    for subtask in get_all_subtasks():
        subtask_tools = get_composite_tools_for_subtask(subtask, refinement_level)
        subtask_counts[subtask] = len(subtask_tools)
    stats["subtask_tool_counts"] = subtask_counts
    
    # Complexity distribution
    complexity_dist = {}
    for count in component_counts:
        key = f"{count}_tools"
        complexity_dist[key] = complexity_dist.get(key, 0) + 1
    stats["complexity_distribution"] = complexity_dist
    
    return stats