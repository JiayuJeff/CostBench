"""
Travel domain refinement step metadata catalog.

Centralizes the refinement step definitions for every subtask and supports dynamic extension.
When a new refinement step is needed, simply modify this file.

Usage:
1. Add a new step: append the new step definition to the corresponding subtask list.
2. Adjust step order: reorder the entries in the list.
3. Update input/output types: edit the ``input_type`` and ``output_type`` fields.

Extension guidelines:
- Every step must include: name, order, input_type, output_type
- ``order`` values must increase consecutively
- The previous step ``output_type`` must match the next step ``input_type``
- Tool names must follow the convention ``{Subtask}_Refinement_Step{order}``
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

REFINEMENT_DIMENSION_MAPPING = {
    0: "",
    1: "availability and seasonal suitability",
    2: "transportation convenience",
    3: "user ratings and reviews",
    4: "policies and regulations",
    5: "cost efficiency",
}

ATOMIC_SEQUENCE_MAPPING = {
    0: "Decide_{task}_Preference, Search_{task}_Candidates, Select_Final_{task}",
    1: "Decide_{task}_Preference, Search_{task}_Candidates, {task}_Refinement_Step1, Select_Final_{task}",
    2: "Decide_{task}_Preference, Search_{task}_Candidates, {task}_Refinement_Step1, {task}_Refinement_Step2, Select_Final_{task}",
    3: "Decide_{task}_Preference, Search_{task}_Candidates, {task}_Refinement_Step1, {task}_Refinement_Step2, {task}_Refinement_Step3, Select_Final_{task}",
    4: "Decide_{task}_Preference, Search_{task}_Candidates, {task}_Refinement_Step1, {task}_Refinement_Step2, {task}_Refinement_Step3, {task}_Refinement_Step4, Select_Final_{task}",
    5: "Decide_{task}_Preference, Search_{task}_Candidates, {task}_Refinement_Step1, {task}_Refinement_Step2, {task}_Refinement_Step3, {task}_Refinement_Step4, {task}_Refinement_Step5, Select_Final_{task}",
}


def get_refinement_dimensions(level: int) -> Dict[int, str]:
    """Return refinement dimension descriptions up to the given level."""

    return {index: desc for index, desc in REFINEMENT_DIMENSION_MAPPING.items() if index <= level}


def get_atomic_sequence(level: int, task: str) -> str:
    """Return the atomic sequence string for the given refinement level."""

    if level not in ATOMIC_SEQUENCE_MAPPING:
        raise ValueError(f"Unsupported refinement level: {level}")
    return ATOMIC_SEQUENCE_MAPPING[level].format(task=task.capitalize())

@dataclass
class RefinementStep:
    """Metadata that describes a refinement step."""
    name: str  # Tool name, e.g. "Location_Refinement_Step1"
    order: int  # Step order, starting at 1
    input_type: str  # Input data type, e.g. "LocationCandidate_Raw"
    output_type: str  # Output data type, e.g. "LocationCandidate_L1_Available"
    description: Optional[str] = None  # Optional human-readable description


# Refinement step directory - central management for all subtasks
REFINEMENT_STEPS_CATALOG = {
    "location": [
        RefinementStep(
            name="Location_Refinement_Step1",
            order=1,
            input_type="LocationCandidate_Raw",
            output_type="LocationCandidate_L1_Available",
            description="Filter by availability/seasonality"
        ),
        RefinementStep(
            name="Location_Refinement_Step2", 
            order=2,
            input_type="LocationCandidate_L1_Available",
            output_type="LocationCandidate_L2_Located",
            description="Filter by location/transportation convenience"
        ),
        RefinementStep(
            name="Location_Refinement_Step3",
            order=3,
            input_type="LocationCandidate_L2_Located",
            output_type="LocationCandidate_L3_Reviewed",
            description="Filter by travel reputation"
        ),
        RefinementStep(
            name="Location_Refinement_Step4",
            order=4,
            input_type="LocationCandidate_L3_Reviewed",
            output_type="LocationCandidate_L4_Verified",
            description="Filter by policy restrictions"
        ),
        RefinementStep(
            name="Location_Refinement_Step5",
            order=5,
            input_type="LocationCandidate_L4_Verified",
            output_type="LocationCandidate_L5_Finalized",
            description="Filter by cost-effectiveness"
        ),
        # 未来可以轻松添加：
        # RefinementStep(
        #     name="Location_Refinement_Step6",
        #     order=6,
        #     input_type="LocationCandidate_L5_Finalized",
        #     output_type="LocationCandidate_L6_Enhanced",
        #     description="Enhance with additional features"
        # ),
    ],
    
    "transportation": [
        RefinementStep(
            name="Transportation_Refinement_Step1",
            order=1,
            input_type="TransportationCandidate_Raw",
            output_type="TransportationCandidate_L1_Available",
            description="Filter by availability/seasonality"
        ),
        RefinementStep(
            name="Transportation_Refinement_Step2",
            order=2,
            input_type="TransportationCandidate_L1_Available",
            output_type="TransportationCandidate_L2_Located",
            description="Filter by departure station accessibility"
        ),
        RefinementStep(
            name="Transportation_Refinement_Step3",
            order=3,
            input_type="TransportationCandidate_L2_Located",
            output_type="TransportationCandidate_L3_Reviewed",
            description="Filter by passenger reputation"
        ),
        RefinementStep(
            name="Transportation_Refinement_Step4",
            order=4,
            input_type="TransportationCandidate_L3_Reviewed",
            output_type="TransportationCandidate_L4_Verified",
            description="Filter by policy restrictions"
        ),
        RefinementStep(
            name="Transportation_Refinement_Step5",
            order=5,
            input_type="TransportationCandidate_L4_Verified",
            output_type="TransportationCandidate_L5_Finalized",
            description="Filter by cost-effectiveness"
        ),
    ],
    
    "accommodation": [
        RefinementStep(
            name="Accommodation_Refinement_Step1",
            order=1,
            input_type="AccommodationCandidate_Raw",
            output_type="AccommodationCandidate_L1_Available",
            description="Filter by availability/seasonality"
        ),
        RefinementStep(
            name="Accommodation_Refinement_Step2",
            order=2,
            input_type="AccommodationCandidate_L1_Available",
            output_type="AccommodationCandidate_L2_Located",
            description="Filter by location convenience"
        ),
        RefinementStep(
            name="Accommodation_Refinement_Step3",
            order=3,
            input_type="AccommodationCandidate_L2_Located",
            output_type="AccommodationCandidate_L3_Reviewed",
            description="Filter by guest reputation"
        ),
        RefinementStep(
            name="Accommodation_Refinement_Step4",
            order=4,
            input_type="AccommodationCandidate_L3_Reviewed",
            output_type="AccommodationCandidate_L4_Verified",
            description="Filter by policy restrictions"
        ),
        RefinementStep(
            name="Accommodation_Refinement_Step5",
            order=5,
            input_type="AccommodationCandidate_L4_Verified",
            output_type="AccommodationCandidate_L5_Finalized",
            description="Filter by cost-effectiveness"
        ),
    ],
    
    "attraction": [
        RefinementStep(
            name="Attraction_Refinement_Step1",
            order=1,
            input_type="AttractionCandidate_Raw",
            output_type="AttractionCandidate_L1_Available",
            description="Filter by availability/seasonality"
        ),
        RefinementStep(
            name="Attraction_Refinement_Step2",
            order=2,
            input_type="AttractionCandidate_L1_Available",
            output_type="AttractionCandidate_L2_Located",
            description="Filter by location convenience"
        ),
        RefinementStep(
            name="Attraction_Refinement_Step3",
            order=3,
            input_type="AttractionCandidate_L2_Located",
            output_type="AttractionCandidate_L3_Reviewed",
            description="Filter by visitor reputation"
        ),
        RefinementStep(
            name="Attraction_Refinement_Step4",
            order=4,
            input_type="AttractionCandidate_L3_Reviewed",
            output_type="AttractionCandidate_L4_Verified",
            description="Filter by policy restrictions"
        ),
        RefinementStep(
            name="Attraction_Refinement_Step5",
            order=5,
            input_type="AttractionCandidate_L4_Verified",
            output_type="AttractionCandidate_L5_Finalized",
            description="Filter by cost-effectiveness"
        ),
    ],
    
    "dining": [
        RefinementStep(
            name="Dining_Refinement_Step1",
            order=1,
            input_type="DiningCandidate_Raw",
            output_type="DiningCandidate_L1_Available",
            description="Filter by availability/seasonality"
        ),
        RefinementStep(
            name="Dining_Refinement_Step2",
            order=2,
            input_type="DiningCandidate_L1_Available",
            output_type="DiningCandidate_L2_Located",
            description="Filter by location convenience"
        ),
        RefinementStep(
            name="Dining_Refinement_Step3",
            order=3,
            input_type="DiningCandidate_L2_Located",
            output_type="DiningCandidate_L3_Reviewed",
            description="Filter by guest reputation"
        ),
        RefinementStep(
            name="Dining_Refinement_Step4",
            order=4,
            input_type="DiningCandidate_L3_Reviewed",
            output_type="DiningCandidate_L4_Verified",
            description="Filter by policy restrictions"
        ),
        RefinementStep(
            name="Dining_Refinement_Step5",
            order=5,
            input_type="DiningCandidate_L4_Verified",
            output_type="DiningCandidate_L5_Finalized",
            description="Filter by cost-effectiveness"
        ),
    ],
    
    "shopping": [
        RefinementStep(
            name="Shopping_Refinement_Step1",
            order=1,
            input_type="ShoppingCandidate_Raw",
            output_type="ShoppingCandidate_L1_Available",
            description="Filter by availability/seasonality"
        ),
        RefinementStep(
            name="Shopping_Refinement_Step2",
            order=2,
            input_type="ShoppingCandidate_L1_Available",
            output_type="ShoppingCandidate_L2_Located",
            description="Filter by location convenience"
        ),
        RefinementStep(
            name="Shopping_Refinement_Step3",
            order=3,
            input_type="ShoppingCandidate_L2_Located",
            output_type="ShoppingCandidate_L3_Reviewed",
            description="Filter by guest reputation"
        ),
        RefinementStep(
            name="Shopping_Refinement_Step4",
            order=4,
            input_type="ShoppingCandidate_L3_Reviewed",
            output_type="ShoppingCandidate_L4_Verified",
            description="Filter by policy restrictions"
        ),
        RefinementStep(
            name="Shopping_Refinement_Step5",
            order=5,
            input_type="ShoppingCandidate_L4_Verified",
            output_type="ShoppingCandidate_L5_Finalized",
            description="Filter by cost-effectiveness"
        ),
    ],
}


def get_refinement_steps_catalog() -> Dict[str, List[RefinementStep]]:
    """Return the complete mapping of subtasks to their refinement steps."""
    return REFINEMENT_STEPS_CATALOG


def get_registered_subtasks() -> List[str]:
    """Return a list of all registered subtask names."""
    return list(REFINEMENT_STEPS_CATALOG.keys())


def get_max_refinement_depth() -> int:
    """Return the maximum refinement depth across all subtasks."""
    max_depth = 0
    for steps in REFINEMENT_STEPS_CATALOG.values():
        if steps:
            max_depth = max(max_depth, max(step.order for step in steps))
    return max_depth


def get_subtask_max_depth(subtask: str) -> int:
    """Return the maximum refinement depth for a given subtask."""
    if subtask not in REFINEMENT_STEPS_CATALOG:
        raise ValueError(f"Unknown subtask: {subtask}")
    
    steps = REFINEMENT_STEPS_CATALOG[subtask]
    if not steps:
        return 0
    
    return max(step.order for step in steps)


def get_refinement_steps_for_subtask(subtask: str, max_order: Optional[int] = None) -> List[RefinementStep]:
    """Return the ordered refinement steps for the given subtask."""
    if subtask not in REFINEMENT_STEPS_CATALOG:
        raise ValueError(f"Unknown subtask: {subtask}")
    
    steps = REFINEMENT_STEPS_CATALOG[subtask]
    
    if max_order is not None:
        steps = [step for step in steps if step.order <= max_order]
    
    return sorted(steps, key=lambda x: x.order)


def get_refinement_step_names_for_subtask(subtask: str, max_order: Optional[int] = None) -> List[str]:
    """Return the refinement step names for the given subtask."""
    steps = get_refinement_steps_for_subtask(subtask, max_order)
    return [step.name for step in steps]


def validate_refinement_chain(subtask: str) -> List[str]:
    """Validate that a subtask's refinement chain is continuous and type-safe."""
    errors = []
    
    if subtask not in REFINEMENT_STEPS_CATALOG:
        errors.append(f"Unknown subtask: {subtask}")
        return errors
    
    steps = get_refinement_steps_for_subtask(subtask)
    
    if not steps:
        errors.append(f"No refinement steps defined for subtask: {subtask}")
        return errors
    
    # Validate consecutive step order
    expected_order = 1
    for step in steps:
        if step.order != expected_order:
            errors.append(f"Step order mismatch in {subtask}: expected {expected_order}, got {step.order}")
        expected_order += 1
    
    # Validate input/output type compatibility
    for i in range(len(steps) - 1):
        current_output = steps[i].output_type
        next_input = steps[i + 1].input_type
        
        if current_output != next_input:
            errors.append(f"Type mismatch in {subtask}: {steps[i].name} outputs '{current_output}' but {steps[i + 1].name} expects '{next_input}'")
    
    return errors


def validate_all_refinement_chains() -> Dict[str, List[str]]:
    """Validate the refinement chains of all subtasks and return any errors found."""
    results = {}
    
    for subtask in get_registered_subtasks():
        errors = validate_refinement_chain(subtask)
        if errors:
            results[subtask] = errors
    
    return results


def get_level_mapping_for_subtask(subtask: str) -> Dict[int, str]:
    """Return a mapping from level index to output type for the given subtask."""
    steps = get_refinement_steps_for_subtask(subtask)
    mapping = {0: f"{subtask.capitalize()}Candidate_Raw"}  # Level 0 maps to the raw candidate type
    
    for step in steps:
        mapping[step.order] = step.output_type
    
    return mapping


def get_extension_guidance(requested_depth: int) -> str:
    """Generate guidance when the requested depth exceeds the registered maximum."""
    max_depth = get_max_refinement_depth()
    
    if requested_depth <= max_depth:
        return f"Requested depth {requested_depth} is within the supported range (max: {max_depth})"
    
    guidance = f"""Requested refinement_level ({requested_depth}) exceeds maximum registered depth ({max_depth}).

To add more refinement steps, please edit: env/domains/travel/refinement_steps_catalog.py

Current registered steps per subtask:"""
    
    for subtask in get_registered_subtasks():
        subtask_max = get_subtask_max_depth(subtask)
        steps = get_refinement_steps_for_subtask(subtask)
        step_names = [step.name for step in steps]
        guidance += f"\n- {subtask}: {subtask_max} steps ({', '.join(step_names)})"
    
    guidance += f"""

Extension example for {subtask}:
1. Add new RefinementStep to the {subtask} list in REFINEMENT_STEPS_CATALOG
2. Define corresponding data types in data_types.py
3. Implement the atomic tool in atomic_tools.py
4. Modify the composite naming and descriptions in composite_tools.py
5. Restart the system

The system will automatically:
- Generate new Select tools for the extended depth
- Create composite tools that include the new steps
- Validate the refinement chain continuity"""
    
    return guidance


if __name__ == "__main__":
    # Test code
    print("=== Refinement Steps Catalog Test ===")
    
    # 1. Basic functionality test
    print(f"Registered subtasks: {get_registered_subtasks()}")
    print(f"Max refinement depth: {get_max_refinement_depth()}")
    
    # 2. Subtask depth test
    for subtask in get_registered_subtasks():
        max_depth = get_subtask_max_depth(subtask)
        print(f"{subtask}: max depth = {max_depth}")
    
    # 3. Step retrieval test
    print(f"\nLocation steps (max_order=3):")
    steps = get_refinement_steps_for_subtask("location", max_order=3)
    for step in steps:
        print(f"  {step.order}: {step.name} ({step.input_type} -> {step.output_type})")
    
    # 4. Validation test
    print(f"\nValidation results:")
    validation_errors = validate_all_refinement_chains()
    if validation_errors:
        for subtask, errors in validation_errors.items():
            print(f"  {subtask}: {errors}")
    else:
        print("  All refinement chains are valid!")
    
    # 5. Extension guidance test
    print(f"\nExtension guidance for depth 8:")
    print(get_extension_guidance(8))
    
    # 6. Level mapping test
    print(f"\nLevel mapping for location:")
    mapping = get_level_mapping_for_subtask("location")
    for level, output_type in mapping.items():
        print(f"  Level {level}: {output_type}")
