"""
Atomic tool definitions for the travel domain.

This module defines all atomic tools used in travel planning, including:
1. Decide_*_Preference tools: create preference objects based on enum options
2. Search_*_Candidates tools: search for candidate objects
3. *_Refinement_Step* tools: refine candidate objects (L1-L5)
4. Select_Final_* tools: choose the final object

Each tool is an atomic operation that cannot be further decomposed.
"""

import sys
import os
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)
from enum import Enum
from typing import List, Dict, Any
from env.domains.travel.enums import *
from env.core.base_types import AtomicTool
from env.core.data_types import (
    # Preference types
    LocationPreference, TransportationPreference, AccommodationPreference,
    AttractionPreference, DiningPreference, ShoppingPreference,
    # Location refinement chain
    LocationCandidate_Raw, LocationCandidate_L1_Available, LocationCandidate_L2_Located,
    LocationCandidate_L3_Reviewed, LocationCandidate_L4_Verified, LocationCandidate_L5_Finalized,
    TravelLocation,
    # Transportation refinement chain
    TransportationCandidate_Raw, TransportationCandidate_L1_Available, 
    TransportationCandidate_L2_Located, TransportationCandidate_L3_Reviewed,
    TransportationCandidate_L4_Verified, TransportationCandidate_L5_Finalized,
    TravelTransportation,
    # Accommodation refinement chain
    AccommodationCandidate_Raw, AccommodationCandidate_L1_Available,
    AccommodationCandidate_L2_Located, AccommodationCandidate_L3_Reviewed,
    AccommodationCandidate_L4_Verified, AccommodationCandidate_L5_Finalized,
    TravelAccommodation,
    # Attraction refinement chain
    AttractionCandidate_Raw, AttractionCandidate_L1_Available,
    AttractionCandidate_L2_Located, AttractionCandidate_L3_Reviewed,
    AttractionCandidate_L4_Verified, AttractionCandidate_L5_Finalized,
    TravelAttraction,
    # Dining refinement chain
    DiningCandidate_Raw, DiningCandidate_L1_Available,
    DiningCandidate_L2_Located, DiningCandidate_L3_Reviewed,
    DiningCandidate_L4_Verified, DiningCandidate_L5_Finalized,
    TravelDining,
    # Shopping refinement chain
    ShoppingCandidate_Raw, ShoppingCandidate_L1_Available,
    ShoppingCandidate_L2_Located, ShoppingCandidate_L3_Reviewed,
    ShoppingCandidate_L4_Verified, ShoppingCandidate_L5_Finalized,
    TravelShopping,
    # Auxiliary types
    TimeInfo, HomeLocation
)

# Retrieve the maximum refinement depth from the metadata directory, avoiding hardcoding
from env.domains.travel.refinement_steps_catalog import get_max_refinement_depth
REFINEMENT_STEPS = get_max_refinement_depth()



# =====================================================
# Decide Preference tools (one per primary task)
# =====================================================

# Location Preference tool
Decide_Location_Preference = AtomicTool(
    name="Decide_Location_Preference",
    input_types=["LocationCategory", "LocationTier", "LocationStyle", "LocationFeaturePackage"],
    output_type="LocationPreference",
    description="With this tool, you describe a location by choosing its category, tier, style, and feature package according to the user requirements. The tool then gives you a unique label that represents your exact combination of location preferences."
)

Decide_Transportation_Preference = AtomicTool(
    name="Decide_Transportation_Preference", 
    input_types=["LocationPreference", "TransportationCategory", "TransportationTier", "TransportationStyle", "TransportationFeaturePackage"],
    output_type="TransportationPreference",
    description="With this tool, you describe a transportation by choosing its category, tier, style, and feature package according to the user requirements. The tool then gives you a unique label that represents your exact combination of transportation preferences."
)

Decide_Accommodation_Preference = AtomicTool(
    name="Decide_Accommodation_Preference",
    input_types=["LocationPreference", "AccommodationCategory", "AccommodationTier", "AccommodationStyle", "AccommodationFeaturePackage"], 
    output_type="AccommodationPreference",
    description="With this tool, you describe an accommodation by choosing its category, tier, style, and feature package according to the user requirements. The tool then gives you a unique label that represents your exact combination of accommodation preferences."
)

Decide_Attraction_Preference = AtomicTool(
    name="Decide_Attraction_Preference",
    input_types=["LocationPreference", "AttractionCategory", "AttractionTier", "AttractionStyle", "AttractionFeaturePackage"],
    output_type="AttractionPreference", 
    description="With this tool, you describe an attraction by choosing its category, tier, style, and feature package according to the user requirements. The tool then gives you a unique label that represents your exact combination of attraction preferences."
)

Decide_Dining_Preference = AtomicTool(
    name="Decide_Dining_Preference",
    input_types=["LocationPreference", "DiningCategory", "DiningTier", "DiningStyle", "DiningFeaturePackage"],
    output_type="DiningPreference",
    description="With this tool, you describe a dining by choosing its category, tier, style, and feature package according to the user requirements. The tool then gives you a unique label that represents your exact combination of dining preferences."
)

Decide_Shopping_Preference = AtomicTool(
    name="Decide_Shopping_Preference", 
    input_types=["LocationPreference", "ShoppingCategory", "ShoppingTier", "ShoppingStyle", "ShoppingFeaturePackage"],
    output_type="ShoppingPreference",
    description="With this tool, you describe a shopping by choosing its category, tier, style, and feature package according to the user requirements. The tool then gives you a unique label that represents your exact combination of shopping preferences."
)


# =====================================================
# Location subtask toolchain
# =====================================================

Search_Location_Candidates = AtomicTool(
    name="Search_Location_Candidates",
    input_types=["LocationPreference", "TimeInfo"],
    output_type="LocationCandidate_Raw",
    description="With this tool, you provide your location preference and a time information. It will then help you find an unfiltered set of all possible locations (represented by a unique id) that match your location preference label."
)

Location_Refinement_Step1 = AtomicTool(
    name="Location_Refinement_Step1", 
    input_types=["LocationCandidate_Raw", "TimeInfo"],
    output_type="LocationCandidate_L1_Available",
    description="This tool takes the unfiltered set of location candidates along with the time, and gives back a filtered set (represented by a unique id) of locations (represented by a unique id) that are available or seasonally suitable at that time."
)

Location_Refinement_Step2 = AtomicTool(
    name="Location_Refinement_Step2",
    input_types=["LocationCandidate_L1_Available"],
    output_type="LocationCandidate_L2_Located", 
    description="This tool takes the location candidate set filtered by availability/seasonality, and further filters to include only those locations (represented by a unique id) that are conveniently accessible from the user’s home location."
)

Location_Refinement_Step3 = AtomicTool(
    name="Location_Refinement_Step3",
    input_types=["LocationCandidate_L2_Located"],
    output_type="LocationCandidate_L3_Reviewed",
    description="This tool takes the location candidate set filtered by availability/seasonality and location/transportation convenience, and further filters to include only those locations (represented by a unique id) that have a good travel reputation."
)

Location_Refinement_Step4 = AtomicTool(
    name="Location_Refinement_Step4",
    input_types=["LocationCandidate_L3_Reviewed"],
    output_type="LocationCandidate_L4_Verified", 
    description="This tool takes the location candidate set filtered by availability/seasonality, location/transportation convenience, and travel reputation, and further filters to include only those locations (represented by a unique id) that have no policy restrictions."
)

Location_Refinement_Step5 = AtomicTool(
    name="Location_Refinement_Step5",
    input_types=["LocationCandidate_L4_Verified"],
    output_type="LocationCandidate_L5_Finalized",
    description="This tool takes the location candidate set filtered by availability/seasonality, location/transportation convenience, travel reputation, and policy restrictions, and further filters to include only those locations (represented by a unique id) that have a high cost-effectiveness."
)

# Select Final Tools are registered on runtime according to the refinement level

# =====================================================
# Transportation subtask toolchain
# =====================================================

Search_Transportation_Candidates = AtomicTool(
    name="Search_Transportation_Candidates",
    input_types=["TransportationPreference", "LocationPreference", "TimeInfo"],
    output_type="TransportationCandidate_Raw",
    description="With this tool, you provide your transportation preference, location preference, and a time information. It will then help you find an unfiltered set of all possible transportation (represented by a unique id) that match your transportation preference label."
)

Transportation_Refinement_Step1 = AtomicTool(
    name="Transportation_Refinement_Step1",
    input_types=["TransportationCandidate_Raw", "TimeInfo"],
    output_type="TransportationCandidate_L1_Available",
    description="This tool takes the unfiltered set of transportation candidates along with the time, and gives back a filtered set (represented by a unique id) of transportation candidates (represented by a unique id) that are available or seasonally suitable at that time."
)

Transportation_Refinement_Step2 = AtomicTool(
    name="Transportation_Refinement_Step2", 
    input_types=["TransportationCandidate_L1_Available"],
    output_type="TransportationCandidate_L2_Located",
    description="This tool takes the unfiltered set of transportation candidates along with the time, and gives back a filtered set (represented by a unique id) of transportation candidates (represented by a unique id) whose departure stations are conveniently accessible from the user’s home location."
)

Transportation_Refinement_Step3 = AtomicTool(
    name="Transportation_Refinement_Step3",
    input_types=["TransportationCandidate_L2_Located"],
    output_type="TransportationCandidate_L3_Reviewed",
    description="This tool takes the transportation candidate set filtered by availability/seasonality and location/transportation convenience, and further filters to include only those transportation candidates (represented by a unique id) that have a good passsager's reputation."
)

Transportation_Refinement_Step4 = AtomicTool(
    name="Transportation_Refinement_Step4",
    input_types=["TransportationCandidate_L3_Reviewed"], 
    output_type="TransportationCandidate_L4_Verified",
    description="This tool takes the transportation candidate set filtered by availability/seasonality, location/transportation convenience, and passsager's reputation, and further filters to include only those transportation candidates (represented by a unique id) that have no policy restrictions."
)

Transportation_Refinement_Step5 = AtomicTool(
    name="Transportation_Refinement_Step5",
    input_types=["TransportationCandidate_L4_Verified"],
    output_type="TransportationCandidate_L5_Finalized",
    description="This tool takes the transportation candidate set filtered by availability/seasonality, location/transportation convenience, passsager's reputation, and policy restrictions, and further filters to include only those transportation candidates (represented by a unique id) that have a high cost-effectiveness."
)

# =====================================================
# Accommodation subtask toolchain
# =====================================================

Search_Accommodation_Candidates = AtomicTool(
    name="Search_Accommodation_Candidates",
    input_types=["AccommodationPreference", "LocationPreference", "TimeInfo"],
    output_type="AccommodationCandidate_Raw",
    description="With this tool, you provide your accommodation preference, location preference, and a time information. It will then help you find an unfiltered set of all possible accommodation (represented by a unique id) that match your accommodation preference label."
)

Accommodation_Refinement_Step1 = AtomicTool(
    name="Accommodation_Refinement_Step1",
    input_types=["AccommodationCandidate_Raw", "TimeInfo"],
    output_type="AccommodationCandidate_L1_Available",
    description="This tool takes the unfiltered set of accommodation candidates along with the time, and gives back a filtered set (represented by a unique id) of accommodation candidates (represented by a unique id) that are available or seasonally suitable at that time."
)

Accommodation_Refinement_Step2 = AtomicTool(
    name="Accommodation_Refinement_Step2",
    input_types=["AccommodationCandidate_L1_Available"],
    output_type="AccommodationCandidate_L2_Located",
    description="This tool takes the accommodation candidate set filtered by availability/seasonality, and further filters to include only those accommodation candidates (represented by a unique id) that the user could conveniently reach."
)

Accommodation_Refinement_Step3 = AtomicTool(
    name="Accommodation_Refinement_Step3",
    input_types=["AccommodationCandidate_L2_Located"],
    output_type="AccommodationCandidate_L3_Reviewed", 
    description="This tool takes the accommodation candidate set filtered by availability/seasonality and location convenience, and further filters to include only those accommodation candidates (represented by a unique id) that have a good guest's reputation."
)

Accommodation_Refinement_Step4 = AtomicTool(
    name="Accommodation_Refinement_Step4",
    input_types=["AccommodationCandidate_L3_Reviewed"],
    output_type="AccommodationCandidate_L4_Verified",
    description="This tool takes the accommodation candidate set filtered by availability/seasonality, location convenience, and guest's reputation, and further filters to include only those accommodation candidates (represented by a unique id) that have no policy restrictions."
)

Accommodation_Refinement_Step5 = AtomicTool(
    name="Accommodation_Refinement_Step5",
    input_types=["AccommodationCandidate_L4_Verified"],
    output_type="AccommodationCandidate_L5_Finalized",
    description="This tool takes the accommodation candidate set filtered by availability/seasonality, location convenience, guest's reputation, and policy restrictions, and further filters to include only those accommodation candidates (represented by a unique id) that have a high cost-effectiveness."
)

# =====================================================
# Attraction subtask toolchain
# =====================================================

Search_Attraction_Candidates = AtomicTool(
    name="Search_Attraction_Candidates",
    input_types=["AttractionPreference", "LocationPreference", "TimeInfo"],
    output_type="AttractionCandidate_Raw",
    description="With this tool, you provide your attraction preference, location preference, and a time information. It will then help you find an unfiltered set of all possible attraction (represented by a unique id) that match your attraction preference label."
)

Attraction_Refinement_Step1 = AtomicTool(
    name="Attraction_Refinement_Step1",
    input_types=["AttractionCandidate_Raw", "TimeInfo"],
    output_type="AttractionCandidate_L1_Available",
    description="This tool takes the unfiltered set of attraction candidates along with the time, and gives back a filtered set (represented by a unique id) of attraction candidates (represented by a unique id) that are available or seasonally suitable at that time."
)

Attraction_Refinement_Step2 = AtomicTool(
    name="Attraction_Refinement_Step2",
    input_types=["AttractionCandidate_L1_Available"],
    output_type="AttractionCandidate_L2_Located",
    description="This tool takes the attraction candidate set filtered by availability/seasonality, and further filters to include only those attraction candidates (represented by a unique id) that the user could conveniently reach."
)

Attraction_Refinement_Step3 = AtomicTool(
    name="Attraction_Refinement_Step3",
    input_types=["AttractionCandidate_L2_Located"],
    output_type="AttractionCandidate_L3_Reviewed",
    description="This tool takes the attraction candidate set filtered by availability/seasonality and location convenience, and further filters to include only those attraction candidates (represented by a unique id) that have a good visitor's reputation."
)

Attraction_Refinement_Step4 = AtomicTool(
    name="Attraction_Refinement_Step4",
    input_types=["AttractionCandidate_L3_Reviewed"],
    output_type="AttractionCandidate_L4_Verified",
    description="This tool takes the attraction candidate set filtered by availability/seasonality, location convenience, and visitor's reputation, and further filters to include only those attraction candidates (represented by a unique id) that have no policy restrictions."
)

Attraction_Refinement_Step5 = AtomicTool(
    name="Attraction_Refinement_Step5",
    input_types=["AttractionCandidate_L4_Verified"],
    output_type="AttractionCandidate_L5_Finalized",
    description="This tool takes the attraction candidate set filtered by availability/seasonality, location convenience, visitor's reputation, and policy restrictions, and further filters to include only those attraction candidates (represented by a unique id) that have a high cost-effectiveness."
)

# =====================================================
# Dining subtask toolchain
# =====================================================

Search_Dining_Candidates = AtomicTool(
    name="Search_Dining_Candidates",
    input_types=["DiningPreference", "LocationPreference", "TimeInfo"],
    output_type="DiningCandidate_Raw",
    description="With this tool, you provide your dining preference, location preference, and a time information. It will then help you find an unfiltered set of all possible dining (represented by a unique id) that match your dining preference label."
)

Dining_Refinement_Step1 = AtomicTool(
    name="Dining_Refinement_Step1",
    input_types=["DiningCandidate_Raw", "TimeInfo"],
    output_type="DiningCandidate_L1_Available",
    description="This tool takes the unfiltered set of dining candidates along with the time, and gives back a filtered set (represented by a unique id) of dining candidates (represented by a unique id) that are available or seasonally suitable at that time."
)

Dining_Refinement_Step2 = AtomicTool(
    name="Dining_Refinement_Step2",
    input_types=["DiningCandidate_L1_Available"],
    output_type="DiningCandidate_L2_Located",
    description="This tool takes the dining candidate set filtered by availability/seasonality, and further filters to include only those dining candidates (represented by a unique id) that the user could conveniently get to."
)

Dining_Refinement_Step3 = AtomicTool(
    name="Dining_Refinement_Step3",
    input_types=["DiningCandidate_L2_Located"],
    output_type="DiningCandidate_L3_Reviewed",
    description="This tool takes the dining candidate set filtered by availability/seasonality and location convenience, and further filters to include only those dining candidates (represented by a unique id) that have a good guest's reputation."
)

Dining_Refinement_Step4 = AtomicTool(
    name="Dining_Refinement_Step4",
    input_types=["DiningCandidate_L3_Reviewed"],
    output_type="DiningCandidate_L4_Verified",
    description="This tool takes the dining candidate set filtered by availability/seasonality, location convenience, and guest's reputation, and further filters to include only those dining candidates (represented by a unique id) that have no policy restrictions."
)

Dining_Refinement_Step5 = AtomicTool(
    name="Dining_Refinement_Step5",
    input_types=["DiningCandidate_L4_Verified"],
    output_type="DiningCandidate_L5_Finalized",
    description="This tool takes the dining candidate set filtered by availability/seasonality, location convenience, guest's reputation, and policy restrictions, and further filters to include only those dining candidates (represented by a unique id) that have a high cost-effectiveness."
)

# =====================================================
# Shopping subtask toolchain
# =====================================================

Search_Shopping_Candidates = AtomicTool(
    name="Search_Shopping_Candidates",
    input_types=["ShoppingPreference", "LocationPreference", "TimeInfo"],
    output_type="ShoppingCandidate_Raw",
    description="With this tool, you provide your shopping preference, location preference, and a time information. It will then help you find an unfiltered set of all possible shopping (represented by a unique id) that match your shopping preference label."
)

Shopping_Refinement_Step1 = AtomicTool(
    name="Shopping_Refinement_Step1",
    input_types=["ShoppingCandidate_Raw", "TimeInfo"],
    output_type="ShoppingCandidate_L1_Available",
    description="This tool takes the unfiltered set of shopping candidates along with the time, and gives back a filtered set (represented by a unique id) of shopping candidates that are available or seasonally suitable at that time."
)

Shopping_Refinement_Step2 = AtomicTool(
    name="Shopping_Refinement_Step2",
    input_types=["ShoppingCandidate_L1_Available"],
    output_type="ShoppingCandidate_L2_Located",
    description="This tool takes the shopping candidate set filtered by availability/seasonality, and further filters to include only those shopping candidates (represented by a unique id) that the user could conveniently get to."
)

Shopping_Refinement_Step3 = AtomicTool(
    name="Shopping_Refinement_Step3",
    input_types=["ShoppingCandidate_L2_Located"],
    output_type="ShoppingCandidate_L3_Reviewed",
    description="This tool takes the shopping candidate set filtered by availability/seasonality and location convenience, and further filters to include only those shopping candidates (represented by a unique id) that have a good guest's reputation."
)

Shopping_Refinement_Step4 = AtomicTool(
    name="Shopping_Refinement_Step4",
    input_types=["ShoppingCandidate_L3_Reviewed"],
    output_type="ShoppingCandidate_L4_Verified",
    description="This tool takes the shopping candidate set filtered by availability/seasonality, location convenience, and guest's reputation, and further filters to include only those shopping candidates (represented by a unique id) that have no policy restrictions."
)

Shopping_Refinement_Step5 = AtomicTool(
    name="Shopping_Refinement_Step5",
    input_types=["ShoppingCandidate_L4_Verified"],
    output_type="ShoppingCandidate_L5_Finalized",
    description="This tool takes the shopping candidate set filtered by availability/seasonality, location convenience, guest's reputation, and policy restrictions, and further filters to include only those shopping candidates (represented by a unique id) that have a high cost-effectiveness."
)

# =====================================================
# Tool registry and helper functions
# =====================================================

# Partial atomic tool registry (excluding all Select_Final tools)
ATOMIC_TOOLS_REGISTRY = {
    # Decide Preference tools
    "Decide_Location_Preference": Decide_Location_Preference,
    "Decide_Transportation_Preference": Decide_Transportation_Preference,
    "Decide_Accommodation_Preference": Decide_Accommodation_Preference,
    "Decide_Attraction_Preference": Decide_Attraction_Preference,
    "Decide_Dining_Preference": Decide_Dining_Preference,
    "Decide_Shopping_Preference": Decide_Shopping_Preference,
    
    # Location toolchain
    "Search_Location_Candidates": Search_Location_Candidates,
    "Location_Refinement_Step1": Location_Refinement_Step1,
    "Location_Refinement_Step2": Location_Refinement_Step2,
    "Location_Refinement_Step3": Location_Refinement_Step3,
    "Location_Refinement_Step4": Location_Refinement_Step4,
    "Location_Refinement_Step5": Location_Refinement_Step5,
    
    # Transportation toolchain
    "Search_Transportation_Candidates": Search_Transportation_Candidates,
    "Transportation_Refinement_Step1": Transportation_Refinement_Step1,
    "Transportation_Refinement_Step2": Transportation_Refinement_Step2,
    "Transportation_Refinement_Step3": Transportation_Refinement_Step3,
    "Transportation_Refinement_Step4": Transportation_Refinement_Step4,
    "Transportation_Refinement_Step5": Transportation_Refinement_Step5,
    
    # Accommodation toolchain
    "Search_Accommodation_Candidates": Search_Accommodation_Candidates,
    "Accommodation_Refinement_Step1": Accommodation_Refinement_Step1,
    "Accommodation_Refinement_Step2": Accommodation_Refinement_Step2,
    "Accommodation_Refinement_Step3": Accommodation_Refinement_Step3,
    "Accommodation_Refinement_Step4": Accommodation_Refinement_Step4,
    "Accommodation_Refinement_Step5": Accommodation_Refinement_Step5,
    
    # Attraction toolchain
    "Search_Attraction_Candidates": Search_Attraction_Candidates,
    "Attraction_Refinement_Step1": Attraction_Refinement_Step1,
    "Attraction_Refinement_Step2": Attraction_Refinement_Step2,
    "Attraction_Refinement_Step3": Attraction_Refinement_Step3,
    "Attraction_Refinement_Step4": Attraction_Refinement_Step4,
    "Attraction_Refinement_Step5": Attraction_Refinement_Step5,
    
    # Dining toolchain
    "Search_Dining_Candidates": Search_Dining_Candidates,
    "Dining_Refinement_Step1": Dining_Refinement_Step1,
    "Dining_Refinement_Step2": Dining_Refinement_Step2,
    "Dining_Refinement_Step3": Dining_Refinement_Step3,
    "Dining_Refinement_Step4": Dining_Refinement_Step4,
    "Dining_Refinement_Step5": Dining_Refinement_Step5,
    
    # Shopping toolchain
    "Search_Shopping_Candidates": Search_Shopping_Candidates,
    "Shopping_Refinement_Step1": Shopping_Refinement_Step1,
    "Shopping_Refinement_Step2": Shopping_Refinement_Step2,
    "Shopping_Refinement_Step3": Shopping_Refinement_Step3,
    "Shopping_Refinement_Step4": Shopping_Refinement_Step4,
    "Shopping_Refinement_Step5": Shopping_Refinement_Step5,
}


def get_decide_preference_tools() -> List[AtomicTool]:
    """Return all Decide_*_Preference tools."""
    return [
        Decide_Location_Preference,
        Decide_Transportation_Preference,
        Decide_Accommodation_Preference,
        Decide_Attraction_Preference,
        Decide_Dining_Preference,
        Decide_Shopping_Preference,
    ]


def get_all_subtasks() -> List[str]:
    """Return all subtask names."""
    return ["location", "transportation", "accommodation", "attraction", "dining", "shopping"]


def get_location_independent_subtasks() -> List[str]:
    """Return subtask names that do not require Location as an input parameter."""
    return ["location"]


def get_location_dependent_subtasks() -> List[str]:
    """Return subtask names that implicitly depend on Location in the query."""  
    return ["transportation", "accommodation", "attraction", "dining", "shopping"]
    
    
def create_dynamic_select_tool(subtask: str, refinement_level: int) -> AtomicTool:
    """Create a Select tool for the specified refinement level."""
    from env.domains.travel.refinement_steps_catalog import get_level_mapping_for_subtask, get_extension_guidance
    
    # Use the metadata directory to obtain the level mapping instead of hardcoding
    level_mapping = get_level_mapping_for_subtask(subtask)
    
    # Verify that the requested level is within the supported range
    if refinement_level not in level_mapping:
        max_depth = max(level_mapping.keys())
        raise ValueError(f"Requested refinement_level ({refinement_level}) exceeds maximum registered depth ({max_depth}) for subtask '{subtask}'.\n{get_extension_guidance(refinement_level)}")
    
    input_type = level_mapping[refinement_level]
    tool_name = f"Select_Final_{subtask.capitalize()}"
    
    return AtomicTool(
        name=tool_name,
        input_types=[input_type],
        output_type=f"Travel{subtask.capitalize()}",
        description=f"With this tool, you can select the single best {subtask} from your final candidate list (after all the screening steps are completed). It takes the most refined set of {subtask} candidates and outputs your finalized travel {subtask} choice."
    )


DYNAMIC_ATOMIC_TOOLS_REGISTRY = {}

def get_all_dynamic_atomic_tools(refinement_level: int) -> Dict[str, AtomicTool]:
    """
    Generate the complete dynamic atomic tool registry.

    Args:
        refinement_level: Refinement level (0-5); all subtasks use the same level.

    Returns:
        Dict[str, AtomicTool]: The full registry containing both static tools and
            dynamically generated Select tools, filtered by refinement level for
            refinement step tools.
    """
    from env.domains.travel.refinement_steps_catalog import get_max_refinement_depth, get_extension_guidance
    
    global DYNAMIC_ATOMIC_TOOLS_REGISTRY
    
    # Verify that the requested level is within the supported range
    max_depth = get_max_refinement_depth()
    if refinement_level > max_depth:
        raise ValueError(f"Requested refinement_level ({refinement_level}) exceeds maximum registered depth ({max_depth}).\n{get_extension_guidance(refinement_level)}")
    
    # Start with the static registry but filter it according to the refinement level
    tools = {}
    
    # Copy all tools that are not refinement steps
    for tool_name, tool in ATOMIC_TOOLS_REGISTRY.items():
        # Check whether this is a refinement step tool
        if "_Refinement_Step" in tool_name:
            # Extract the step number
            try:
                step_num = int(tool_name.split("_Refinement_Step")[1])
                # Include only the steps allowed by the current refinement level
                if step_num <= refinement_level:
                    tools[tool_name] = tool
            except (IndexError, ValueError):
                # If parsing fails, include the tool to be safe
                tools[tool_name] = tool
        else:
            # Non-refinement step tools are included directly
            tools[tool_name] = tool
    
    # Add a Select tool for each subtask at the given level
    for subtask in get_all_subtasks():
        select_tool = create_dynamic_select_tool(subtask, refinement_level)
        tools[select_tool.name] = select_tool
    
    # Update the global registry
    DYNAMIC_ATOMIC_TOOLS_REGISTRY = tools
    
    return tools


def get_dynamic_atomic_tool(tool_name: str, refinement_level: int) -> AtomicTool:
    """
    Retrieve a specific dynamic atomic tool.

    Args:
        tool_name: Name of the tool.
        refinement_level: Refinement level (0-5).

    Returns:
        AtomicTool: The requested tool instance.
    """
    # Generate the complete dynamic tool registry first
    all_tools = get_all_dynamic_atomic_tools(refinement_level)
    
    # Look up the requested tool
    if tool_name not in all_tools:
        raise ValueError(f"Tool '{tool_name}' not found in dynamic registry")
    
    return all_tools[tool_name]


def get_tools_by_subtask(subtask: str, refinement_level: int) -> List[AtomicTool]:
    """Retrieve all tools for the given subtask, generated dynamically from the metadata catalog."""
    from env.domains.travel.refinement_steps_catalog import get_refinement_step_names_for_subtask, get_extension_guidance
    
    # ✅ Retrieve the dynamic tool registry for this level
    all_tools = get_all_dynamic_atomic_tools(refinement_level)
    
    subtask_cap = subtask.capitalize()
    
    # ✅ Build the tool name list based on the metadata catalog instead of hardcoding
    expected_tool_names = []
    
    # Base tools
    expected_tool_names.append(f"Decide_{subtask_cap}_Preference")
    expected_tool_names.append(f"Search_{subtask_cap}_Candidates")
    
    # Add the corresponding refinement steps based on the metadata catalog
    try:
        refinement_step_names = get_refinement_step_names_for_subtask(subtask, max_order=refinement_level)
        expected_tool_names.extend(refinement_step_names)
    except ValueError as e:
        # Provide guidance on extending the catalog if the requested level exceeds the registered depth
        raise ValueError(f"Error getting refinement steps for subtask '{subtask}' at level {refinement_level}: {e}\n{get_extension_guidance(refinement_level)}")
    
    # Select tools
    expected_tool_names.append(f"Select_Final_{subtask_cap}")
    
    # ✅ Only return tools that actually exist in the dynamic registry
    subtask_tools = []
    for tool_name in expected_tool_names:
        if tool_name in all_tools:  # This check is critical!
            subtask_tools.append(all_tools[tool_name])
    
    return subtask_tools


# Enum mapping table (for agent reference)
ENUM_MAPPINGS = {
    "LocationCategory": LocationCategory,
    "LocationTier": LocationTier, 
    "LocationStyle": LocationStyle,
    "LocationFeaturePackage": LocationFeaturePackage,
    "TransportationCategory": TransportationCategory,
    "TransportationTier": TransportationTier,
    "TransportationStyle": TransportationStyle,
    "TransportationFeaturePackage": TransportationFeaturePackage,
    "AccommodationCategory": AccommodationCategory,
    "AccommodationTier": AccommodationTier,
    "AccommodationStyle": AccommodationStyle,
    "AccommodationFeaturePackage": AccommodationFeaturePackage,
    "AttractionCategory": AttractionCategory,
    "AttractionTier": AttractionTier,
    "AttractionStyle": AttractionStyle,
    "AttractionFeaturePackage": AttractionFeaturePackage,
    "DiningCategory": DiningCategory,
    "DiningTier": DiningTier,
    "DiningStyle": DiningStyle,
    "DiningFeaturePackage": DiningFeaturePackage,
    "ShoppingCategory": ShoppingCategory,
    "ShoppingTier": ShoppingTier,
    "ShoppingStyle": ShoppingStyle,
    "ShoppingFeaturePackage": ShoppingFeaturePackage,
}


def get_enum_options(enum_name: str) -> List[str]:
    """
    Retrieve all option values for the specified enum.

    Args:
        enum_name: Enum name, such as "LocationCategory".

    Returns:
        List[str]: List of enum values.
    """
    if enum_name not in ENUM_MAPPINGS:
        raise ValueError(f"Unknown enum: {enum_name}")
    
    enum_class = ENUM_MAPPINGS[enum_name]
    return [e.value for e in enum_class]


if __name__ == "__main__":
    print("=== Dynamic Atomic Tools Test ===")
    
    # Display the number of static tools
    print(f"Static atomic tools: {len(ATOMIC_TOOLS_REGISTRY)}")
    
    # 1. Test dynamic tool registry generation
    print(f"\n{'='*50}")
    print("1. DYNAMIC TOOLS REGISTRY TEST")
    print(f"{'='*50}")
    
    test_levels = [0, 2, 5]
    
    for level in test_levels:
        print(f"\n--- Testing refinement level L{level} ---")
        
        # Generate the dynamic tool registry
        dynamic_tools = get_all_dynamic_atomic_tools(level)
        static_count = len(ATOMIC_TOOLS_REGISTRY)
        select_count = len(get_all_subtasks())  # One Select tool per subtask
        total_count = len(dynamic_tools)
        
        print(f"L{level} registry: {static_count} static + {select_count} select = {total_count} total tools")
        
        # Display the generated Select tools
        select_tools = [name for name in dynamic_tools.keys() if name.startswith("Select_Final_")]
        print(f"Generated Select tools: {select_tools}")
        
        # Verify the input type for each Select tool
        for subtask in get_all_subtasks():
            select_tool = dynamic_tools[f"Select_Final_{subtask.capitalize()}"]
            print(f"  {select_tool.name}: {select_tool.input_types[0]} -> {select_tool.output_type}")
    
    # 2. Test single tool lookup
    print(f"\n{'='*50}")
    print("2. SINGLE TOOL LOOKUP TEST")
    print(f"{'='*50}")
    
    test_tools = ["Select_Final_Location", "Location_Refinement_Step3", "Decide_Transportation_Preference"]
    refinement_level = 3
    
    print(f"Testing tool lookup with refinement level L{refinement_level}:")
    
    for tool_name in test_tools:
        try:
            tool = get_dynamic_atomic_tool(tool_name, refinement_level)
            print(f"✅ {tool_name}: {tool.input_types} -> {tool.output_type}")
        except ValueError as e:
            print(f"❌ {tool_name}: {e}")
    
    # 3. Test subtask tool filtering
    print(f"\n{'='*50}")
    print("3. SUBTASK TOOLS FILTERING TEST") 
    print(f"{'='*50}")
    
    test_subtasks = ["location", "transportation", "dining"]
    refinement_level = 4
    
    print(f"Testing subtask filtering with refinement level L{refinement_level}:")
    
    for subtask in test_subtasks:
        tools = get_tools_by_subtask(subtask, refinement_level)
        print(f"\n{subtask.upper()} subtask tools ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool.name}: {tool.input_types} -> {tool.output_type}")
    
    # 4. Test Select tool input types at different levels
    print(f"\n{'='*50}")
    print("4. SELECT TOOL INPUT TYPES TEST")
    print(f"{'='*50}")
    
    subtask = "accommodation"
    
    print(f"Testing {subtask} Select tool at different refinement levels:")
    
    for level in range(6):  # L0 to L5
        select_tool = get_dynamic_atomic_tool(f"Select_Final_{subtask.capitalize()}", level)
        input_type = select_tool.input_types[0]
        print(f"  L{level}: {select_tool.name} accepts {input_type}")
    
    # 5. Test benchmark scenarios
    print(f"\n{'='*50}")
    print("5. BENCHMARK SCENARIOS TEST")
    print(f"{'='*50}")
    
    scenarios = {
        "Fast Decision": 1,
        "Balanced Quality": 3, 
        "Premium Quality": 5
    }
    
    for scenario_name, level in scenarios.items():
        print(f"\n--- {scenario_name} (L{level}) ---")
        
        # Generate the tool registry for this level
        tools = get_all_dynamic_atomic_tools(level)
        
        # Statistics
        total_tools = len(tools)
        select_tools = [name for name in tools.keys() if name.startswith("Select_Final_")]
        
        print(f"Total tools: {total_tools}")
        print(f"Select tools: {len(select_tools)}")
        
        # Display the toolchain length for each subtask
        print("Tool chain lengths:")
        for subtask in get_all_subtasks():
            subtask_tools = get_tools_by_subtask(subtask, level)
            print(f"  {subtask:15s}: {len(subtask_tools)} tools")
        
        # Display the input requirements for each Select tool
        print("Select tools input requirements:")
        for subtask in get_all_subtasks():
            select_tool = tools[f"Select_Final_{subtask.capitalize()}"]
            input_type = select_tool.input_types[0]
            level_suffix = input_type.split('_')[-1] if '_L' in input_type else "Raw"
            print(f"  {subtask:15s}: requires {level_suffix} level input")
    
    # 6. Summary
    print(f"\n{'='*50}")
    print("6. SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Static tools: {len(ATOMIC_TOOLS_REGISTRY)}")
    print(f"✅ Subtasks: {get_all_subtasks()}")
    print(f"✅ Refinement levels: 0 (Raw) to 5 (Finalized)")
    print(f"✅ Dynamic Select tool generation: Working")
    print(f"✅ Single tool lookup: Working") 
    print(f"✅ Subtask filtering: Working")
    print(f"✅ All refinement levels supported: 0-5")
    
    print(f"\n🎯 Dynamic atomic tools system ready for use!")