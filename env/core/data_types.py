"""
Definition of all data types

Defines every concrete data type used in the travel planning system, including preference types,
candidate object upgrade chains, finalized types, and more.

Data type upgrade chain pattern:
- Raw: original search results
- L1-L5: progressively refined candidates (L stands for Level)
- Finalized: candidates that have completed refinement
- Final: the object ultimately chosen by the user
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import sys
import os
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
from env.core.base_types import DataType, BasePreference
from env.utils.id_generator import get_global_generator
from env.domains.travel.enums import *


# =====================================================
# Specific preference types
# =====================================================

@dataclass
class LocationPreference(BasePreference):
    def get_type_name(self) -> str:
        return "LocationPreference"


@dataclass
class TransportationPreference(BasePreference):
    def get_type_name(self) -> str:
        return "TransportationPreference"


@dataclass
class AccommodationPreference(BasePreference):
    def get_type_name(self) -> str:
        return "AccommodationPreference"


@dataclass
class AttractionPreference(BasePreference):
    def get_type_name(self) -> str:
        return "AttractionPreference"


@dataclass
class DiningPreference(BasePreference):
    def get_type_name(self) -> str:
        return "DiningPreference"


@dataclass
class ShoppingPreference(BasePreference):
    def get_type_name(self) -> str:
        return "ShoppingPreference"


# =====================================================
# Candidate base classes (unified structure, simplified version)
# =====================================================

@dataclass
class BaseCandidate(DataType, ABC):
    """Abstract base class for every candidate object"""
    unique_id: str
    previous_candidate_id: str    # ID of the previous-level candidate
    previous_candidate: 'BaseCandidate'  # Previous-level candidate instance
    
    def get_unique_id(self) -> str:
        return self.unique_id
    
    @abstractmethod
    def get_type_name(self) -> str:
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unique_id": self.unique_id,
            "previous_candidate_id": self.previous_candidate_id,
            "previous_candidate": self.previous_candidate.to_dict() if self.previous_candidate else None
        }
    
    @classmethod
    def create_from_previous(cls, previous_candidate: 'BaseCandidate') -> 'BaseCandidate':
        """Create the current level candidate from the previous level candidate"""
        id_gen = get_global_generator()
        unique_id = id_gen.generate_id(cls.__name__)
        return cls(unique_id, previous_candidate.unique_id, previous_candidate)


@dataclass(init=False)
class BaseCandidateRaw(BaseCandidate):
    """Base class for raw candidates (Raw level, no preceding candidate)"""
    preference_id: str
    preference: BasePreference
    
    def __init__(self, unique_id: str, preference_id: str, preference: BasePreference):
        # Raw-level candidates have no preceding candidate
        super().__init__(unique_id, "", None)
        self.preference_id = preference_id
        self.preference = preference
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unique_id": self.unique_id,
            "preference_id": self.preference_id,
            "preference": self.preference.to_dict()
        }
    
    @classmethod
    def create_from_preference(cls, preference: BasePreference) -> 'BaseCandidateRaw':
        id_gen = get_global_generator()
        unique_id = id_gen.generate_id(cls.__name__)
        print("preference unique_id:", preference.unique_id)
        return cls(unique_id, preference.unique_id, preference)


# =====================================================
# Location upgrade chain (full version)
# =====================================================

@dataclass(init=False)
class LocationCandidate_Raw(BaseCandidateRaw):
    """Raw location candidate"""
    
    def get_type_name(self) -> str:
        return "LocationCandidate_Raw"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LocationCandidate_Raw':
        preference = LocationPreference.from_dict(data["preference"])
        return cls(data["unique_id"], data["preference_id"], preference)


@dataclass
class LocationCandidate_L1_Available(BaseCandidate):
    """Level 1: Location candidate after verifying season and timing suitability"""
    
    def get_type_name(self) -> str:
        return "LocationCandidate_L1_Available"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LocationCandidate_L1_Available':
        previous_candidate = LocationCandidate_Raw.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass  
class LocationCandidate_L2_Located(BaseCandidate):
    """Level 2: Location candidate after analyzing transportation convenience"""
    
    def get_type_name(self) -> str:
        return "LocationCandidate_L2_Located"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LocationCandidate_L2_Located':
        previous_candidate = LocationCandidate_L1_Available.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class LocationCandidate_L3_Reviewed(BaseCandidate):
    """Level 3: Location candidate after reviewing travel reputation and guides"""
    
    def get_type_name(self) -> str:
        return "LocationCandidate_L3_Reviewed"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LocationCandidate_L3_Reviewed':
        previous_candidate = LocationCandidate_L2_Located.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class LocationCandidate_L4_Verified(BaseCandidate):
    """Level 4: Location candidate after verifying visa and travel restriction policies"""
    
    def get_type_name(self) -> str:
        return "LocationCandidate_L4_Verified"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LocationCandidate_L4_Verified':
        previous_candidate = LocationCandidate_L3_Reviewed.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class LocationCandidate_L5_Finalized(BaseCandidate):
    """Level 5: Location candidate after estimating total travel budget and value for money"""
    
    def get_type_name(self) -> str:
        return "LocationCandidate_L5_Finalized"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LocationCandidate_L5_Finalized':
        previous_candidate = LocationCandidate_L4_Verified.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class TravelLocation(BaseCandidate):
    """Finalized travel location selection"""
    
    def get_type_name(self) -> str:
        return "TravelLocation"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], refinement_level: int = 5) -> 'TravelLocation':
        """
        Create a `TravelLocation` instance from a dictionary

        Args:
            data: data dictionary
            refinement_level: refinement level (0-5) determining which candidate level to use
        """
        previous_candidate_data = data["previous_candidate"]
        
        # Determine which candidate type to instantiate based on the refinement level
        level_mapping = {
            0: LocationCandidate_Raw,
            1: LocationCandidate_L1_Available,
            2: LocationCandidate_L2_Located,
            3: LocationCandidate_L3_Reviewed,
            4: LocationCandidate_L4_Verified,
            5: LocationCandidate_L5_Finalized
        }
        
        if refinement_level not in level_mapping:
            raise ValueError(f"Invalid refinement level: {refinement_level}. Must be 0-5")
        
        candidate_class = level_mapping[refinement_level]
        previous_candidate = candidate_class.from_dict(previous_candidate_data)
        
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


# =====================================================
# Transportation upgrade chain (full version)
# =====================================================

@dataclass(init=False)
class TransportationCandidate_Raw(BaseCandidateRaw):
    """Raw transportation candidate"""
    
    def get_type_name(self) -> str:
        return "TransportationCandidate_Raw"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransportationCandidate_Raw':
        preference = TransportationPreference.from_dict(data["preference"])
        return cls(data["unique_id"], data["preference_id"], preference)


@dataclass
class TransportationCandidate_L1_Available(BaseCandidate):
    """Level 1: Transportation candidate after checking schedule availability"""
    
    def get_type_name(self) -> str:
        return "TransportationCandidate_L1_Available"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransportationCandidate_L1_Available':
        previous_candidate = TransportationCandidate_Raw.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class TransportationCandidate_L2_Located(BaseCandidate):
    """Level 2: Transportation candidate after analyzing routes and distances"""
    
    def get_type_name(self) -> str:
        return "TransportationCandidate_L2_Located"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransportationCandidate_L2_Located':
        previous_candidate = TransportationCandidate_L1_Available.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class TransportationCandidate_L3_Reviewed(BaseCandidate):
    """Level 3: Transportation candidate after reviewing ratings and service quality"""
    
    def get_type_name(self) -> str:
        return "TransportationCandidate_L3_Reviewed"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransportationCandidate_L3_Reviewed':
        previous_candidate = TransportationCandidate_L2_Located.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class TransportationCandidate_L4_Verified(BaseCandidate):
    """Level 4: Transportation candidate after verifying ticket policies and restrictions"""
    
    def get_type_name(self) -> str:
        return "TransportationCandidate_L4_Verified"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransportationCandidate_L4_Verified':
        previous_candidate = TransportationCandidate_L3_Reviewed.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class TransportationCandidate_L5_Finalized(BaseCandidate):
    """Level 5: Transportation candidate after estimating costs and value for money"""
    
    def get_type_name(self) -> str:
        return "TransportationCandidate_L5_Finalized"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransportationCandidate_L5_Finalized':
        previous_candidate = TransportationCandidate_L4_Verified.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class TravelTransportation(BaseCandidate):
    """Finalized travel transportation selection"""
    
    def get_type_name(self) -> str:
        return "TravelTransportation"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], refinement_level: int = 5) -> 'TravelTransportation':
        previous_candidate_data = data["previous_candidate"]
        
        level_mapping = {
            0: TransportationCandidate_Raw,
            1: TransportationCandidate_L1_Available,
            2: TransportationCandidate_L2_Located,
            3: TransportationCandidate_L3_Reviewed,
            4: TransportationCandidate_L4_Verified,
            5: TransportationCandidate_L5_Finalized
        }
        
        candidate_class = level_mapping[refinement_level]
        previous_candidate = candidate_class.from_dict(previous_candidate_data)
        
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


# =====================================================
# Accommodation upgrade chain (full version)
# =====================================================

@dataclass(init=False)
class AccommodationCandidate_Raw(BaseCandidateRaw):
    """Raw accommodation candidate"""
    
    def get_type_name(self) -> str:
        return "AccommodationCandidate_Raw"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccommodationCandidate_Raw':
        preference = AccommodationPreference.from_dict(data["preference"])
        return cls(data["unique_id"], data["preference_id"], preference)


@dataclass
class AccommodationCandidate_L1_Available(BaseCandidate):
    """Level 1: Accommodation candidate after checking room availability"""
    
    def get_type_name(self) -> str:
        return "AccommodationCandidate_L1_Available"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccommodationCandidate_L1_Available':
        previous_candidate = AccommodationCandidate_Raw.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class AccommodationCandidate_L2_Located(BaseCandidate):
    """Level 2: Accommodation candidate after analyzing location convenience"""
    
    def get_type_name(self) -> str:
        return "AccommodationCandidate_L2_Located"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccommodationCandidate_L2_Located':
        previous_candidate = AccommodationCandidate_L1_Available.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class AccommodationCandidate_L3_Reviewed(BaseCandidate):
    """Level 3: Accommodation candidate after reviewing ratings and facility quality"""
    
    def get_type_name(self) -> str:
        return "AccommodationCandidate_L3_Reviewed"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccommodationCandidate_L3_Reviewed':
        previous_candidate = AccommodationCandidate_L2_Located.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class AccommodationCandidate_L4_Verified(BaseCandidate):
    """Level 4: Accommodation candidate after verifying booking policies and cancellation terms"""
    
    def get_type_name(self) -> str:
        return "AccommodationCandidate_L4_Verified"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccommodationCandidate_L4_Verified':
        previous_candidate = AccommodationCandidate_L3_Reviewed.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class AccommodationCandidate_L5_Finalized(BaseCandidate):
    """Level 5: Accommodation candidate after estimating lodging costs and value for money"""
    
    def get_type_name(self) -> str:
        return "AccommodationCandidate_L5_Finalized"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccommodationCandidate_L5_Finalized':
        previous_candidate = AccommodationCandidate_L4_Verified.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class TravelAccommodation(BaseCandidate):
    """Finalized travel accommodation selection"""
    
    def get_type_name(self) -> str:
        return "TravelAccommodation"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], refinement_level: int = 5) -> 'TravelAccommodation':
        previous_candidate_data = data["previous_candidate"]
        
        level_mapping = {
            0: AccommodationCandidate_Raw,
            1: AccommodationCandidate_L1_Available,
            2: AccommodationCandidate_L2_Located,
            3: AccommodationCandidate_L3_Reviewed,
            4: AccommodationCandidate_L4_Verified,
            5: AccommodationCandidate_L5_Finalized
        }
        
        candidate_class = level_mapping[refinement_level]
        previous_candidate = candidate_class.from_dict(previous_candidate_data)
        
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


# =====================================================
# Attraction upgrade chain (full version)
# =====================================================

@dataclass(init=False)
class AttractionCandidate_Raw(BaseCandidateRaw):
    """Raw attraction candidate"""
    
    def get_type_name(self) -> str:
        return "AttractionCandidate_Raw"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttractionCandidate_Raw':
        preference = AttractionPreference.from_dict(data["preference"])
        return cls(data["unique_id"], data["preference_id"], preference)


@dataclass
class AttractionCandidate_L1_Available(BaseCandidate):
    """Level 1: Attraction candidate after checking opening hours and seasonality"""
    
    def get_type_name(self) -> str:
        return "AttractionCandidate_L1_Available"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttractionCandidate_L1_Available':
        previous_candidate = AttractionCandidate_Raw.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class AttractionCandidate_L2_Located(BaseCandidate):
    """Level 2: Attraction candidate after analyzing location and transportation convenience"""
    
    def get_type_name(self) -> str:
        return "AttractionCandidate_L2_Located"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttractionCandidate_L2_Located':
        previous_candidate = AttractionCandidate_L1_Available.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class AttractionCandidate_L3_Reviewed(BaseCandidate):
    """Level 3: Attraction candidate after reviewing visitor ratings and recommendation indices"""
    
    def get_type_name(self) -> str:
        return "AttractionCandidate_L3_Reviewed"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttractionCandidate_L3_Reviewed':
        previous_candidate = AttractionCandidate_L2_Located.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class AttractionCandidate_L4_Verified(BaseCandidate):
    """Level 4: Attraction candidate after verifying ticket policies and reservation requirements"""
    
    def get_type_name(self) -> str:
        return "AttractionCandidate_L4_Verified"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttractionCandidate_L4_Verified':
        previous_candidate = AttractionCandidate_L3_Reviewed.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class AttractionCandidate_L5_Finalized(BaseCandidate):
    """Level 5: Attraction candidate after estimating visiting costs and schedule planning"""
    
    def get_type_name(self) -> str:
        return "AttractionCandidate_L5_Finalized"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttractionCandidate_L5_Finalized':
        previous_candidate = AttractionCandidate_L4_Verified.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class TravelAttraction(BaseCandidate):
    """Finalized travel attraction selection"""
    
    def get_type_name(self) -> str:
        return "TravelAttraction"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], refinement_level: int = 5) -> 'TravelAttraction':
        previous_candidate_data = data["previous_candidate"]
        
        level_mapping = {
            0: AttractionCandidate_Raw,
            1: AttractionCandidate_L1_Available,
            2: AttractionCandidate_L2_Located,
            3: AttractionCandidate_L3_Reviewed,
            4: AttractionCandidate_L4_Verified,
            5: AttractionCandidate_L5_Finalized
        }
        
        candidate_class = level_mapping[refinement_level]
        previous_candidate = candidate_class.from_dict(previous_candidate_data)
        
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


# =====================================================
# Dining upgrade chain (full version)
# =====================================================

@dataclass(init=False)
class DiningCandidate_Raw(BaseCandidateRaw):
    """Raw dining candidate"""
    
    def get_type_name(self) -> str:
        return "DiningCandidate_Raw"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiningCandidate_Raw':
        preference = DiningPreference.from_dict(data["preference"])
        return cls(data["unique_id"], data["preference_id"], preference)


@dataclass
class DiningCandidate_L1_Available(BaseCandidate):
    """Level 1: Dining candidate after checking business hours and reservation availability"""
    
    def get_type_name(self) -> str:
        return "DiningCandidate_L1_Available"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiningCandidate_L1_Available':
        previous_candidate = DiningCandidate_Raw.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class DiningCandidate_L2_Located(BaseCandidate):
    """Level 2: Dining candidate after analyzing location and convenience"""
    
    def get_type_name(self) -> str:
        return "DiningCandidate_L2_Located"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiningCandidate_L2_Located':
        previous_candidate = DiningCandidate_L1_Available.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class DiningCandidate_L3_Reviewed(BaseCandidate):
    """Level 3: Dining candidate after reviewing dish ratings and service quality"""
    
    def get_type_name(self) -> str:
        return "DiningCandidate_L3_Reviewed"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiningCandidate_L3_Reviewed':
        previous_candidate = DiningCandidate_L2_Located.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class DiningCandidate_L4_Verified(BaseCandidate):
    """Level 4: Dining candidate after verifying reservation policies and special requirements"""
    
    def get_type_name(self) -> str:
        return "DiningCandidate_L4_Verified"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiningCandidate_L4_Verified':
        previous_candidate = DiningCandidate_L3_Reviewed.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class DiningCandidate_L5_Finalized(BaseCandidate):
    """Level 5: Dining candidate after estimating meal costs and value for money"""
    
    def get_type_name(self) -> str:
        return "DiningCandidate_L5_Finalized"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiningCandidate_L5_Finalized':
        previous_candidate = DiningCandidate_L4_Verified.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class TravelDining(BaseCandidate):
    """Finalized travel dining selection"""
    
    def get_type_name(self) -> str:
        return "TravelDining"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], refinement_level: int = 5) -> 'TravelDining':
        previous_candidate_data = data["previous_candidate"]
        
        level_mapping = {
            0: DiningCandidate_Raw,
            1: DiningCandidate_L1_Available,
            2: DiningCandidate_L2_Located,
            3: DiningCandidate_L3_Reviewed,
            4: DiningCandidate_L4_Verified,
            5: DiningCandidate_L5_Finalized
        }
        
        candidate_class = level_mapping[refinement_level]
        previous_candidate = candidate_class.from_dict(previous_candidate_data)
        
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


# =====================================================
# Shopping upgrade chain (full version)
# =====================================================

@dataclass(init=False)
class ShoppingCandidate_Raw(BaseCandidateRaw):
    """Raw shopping candidate"""
    
    def get_type_name(self) -> str:
        return "ShoppingCandidate_Raw"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShoppingCandidate_Raw':
        preference = ShoppingPreference.from_dict(data["preference"])
        return cls(data["unique_id"], data["preference_id"], preference)


@dataclass
class ShoppingCandidate_L1_Available(BaseCandidate):
    """Level 1: Shopping candidate after checking business hours and product availability"""
    
    def get_type_name(self) -> str:
        return "ShoppingCandidate_L1_Available"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShoppingCandidate_L1_Available':
        previous_candidate = ShoppingCandidate_Raw.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class ShoppingCandidate_L2_Located(BaseCandidate):
    """Level 2: Shopping candidate after analyzing location and transportation convenience"""
    
    def get_type_name(self) -> str:
        return "ShoppingCandidate_L2_Located"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShoppingCandidate_L2_Located':
        previous_candidate = ShoppingCandidate_L1_Available.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class ShoppingCandidate_L3_Reviewed(BaseCandidate):
    """Level 3: Shopping candidate after reviewing product quality and price evaluations"""
    
    def get_type_name(self) -> str:
        return "ShoppingCandidate_L3_Reviewed"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShoppingCandidate_L3_Reviewed':
        previous_candidate = ShoppingCandidate_L2_Located.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class ShoppingCandidate_L4_Verified(BaseCandidate):
    """Level 4: Shopping candidate after verifying return policies and payment methods"""
    
    def get_type_name(self) -> str:
        return "ShoppingCandidate_L4_Verified"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShoppingCandidate_L4_Verified':
        previous_candidate = ShoppingCandidate_L3_Reviewed.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class ShoppingCandidate_L5_Finalized(BaseCandidate):
    """Level 5: Shopping candidate after estimating shopping costs and budget allocation"""
    
    def get_type_name(self) -> str:
        return "ShoppingCandidate_L5_Finalized"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShoppingCandidate_L5_Finalized':
        previous_candidate = ShoppingCandidate_L4_Verified.from_dict(data["previous_candidate"])
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


@dataclass
class TravelShopping(BaseCandidate):
    """Finalized travel shopping selection"""
    
    def get_type_name(self) -> str:
        return "TravelShopping"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], refinement_level: int = 5) -> 'TravelShopping':
        previous_candidate_data = data["previous_candidate"]
        
        level_mapping = {
            0: ShoppingCandidate_Raw,
            1: ShoppingCandidate_L1_Available,
            2: ShoppingCandidate_L2_Located,
            3: ShoppingCandidate_L3_Reviewed,
            4: ShoppingCandidate_L4_Verified,
            5: ShoppingCandidate_L5_Finalized
        }
        
        candidate_class = level_mapping[refinement_level]
        previous_candidate = candidate_class.from_dict(previous_candidate_data)
        
        return cls(data["unique_id"], data["previous_candidate_id"], previous_candidate)


# =====================================================
# Auxiliary types
# =====================================================

@dataclass
class TimeInfo(DataType):
    """Time information"""
    unique_id: str
    departure_time_id: str        # Departure time ID, e.g., <TimeInfo00001>
    return_time_id: str          # Return time ID, e.g., <TimeInfo00002>
    
    def get_unique_id(self) -> str:
        return self.unique_id
    
    def get_type_name(self) -> str:
        return "TimeInfo"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeInfo':
        return cls(**data)
    
    @classmethod
    def create(cls, departure_time_id: str, return_time_id: str) -> 'TimeInfo':
        id_gen = get_global_generator()
        unique_id = id_gen.generate_id("TimeInfo")
        return cls(unique_id, departure_time_id, return_time_id)


@dataclass
class HomeLocation(DataType):
    """Home departure information"""
    unique_id: str
    location_description: str     # Location description
    
    def get_unique_id(self) -> str:
        return self.unique_id
    
    def get_type_name(self) -> str:
        return "HomeLocation"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HomeLocation':
        return cls(**data)
    
    @classmethod
    def create(cls, location_description: str) -> 'HomeLocation':
        id_gen = get_global_generator()
        unique_id = id_gen.generate_id("HomeLocation")
        return cls(unique_id, location_description)


# =====================================================
# Type mapping and utility helpers
# =====================================================

DATA_TYPE_REGISTRY = {
    # Preference types
    "LocationPreference": LocationPreference,
    "TransportationPreference": TransportationPreference,
    "AccommodationPreference": AccommodationPreference,
    "AttractionPreference": AttractionPreference,
    "DiningPreference": DiningPreference,
    "ShoppingPreference": ShoppingPreference,
    
    # Location upgrade chain
    "LocationCandidate_Raw": LocationCandidate_Raw,
    "LocationCandidate_L1_Available": LocationCandidate_L1_Available,
    "LocationCandidate_L2_Located": LocationCandidate_L2_Located,
    "LocationCandidate_L3_Reviewed": LocationCandidate_L3_Reviewed,
    "LocationCandidate_L4_Verified": LocationCandidate_L4_Verified,
    "LocationCandidate_L5_Finalized": LocationCandidate_L5_Finalized,
    "TravelLocation": TravelLocation,
    
    # Transportation upgrade chain
    "TransportationCandidate_Raw": TransportationCandidate_Raw,
    "TransportationCandidate_L1_Available": TransportationCandidate_L1_Available,
    "TransportationCandidate_L2_Located": TransportationCandidate_L2_Located,
    "TransportationCandidate_L3_Reviewed": TransportationCandidate_L3_Reviewed,
    "TransportationCandidate_L4_Verified": TransportationCandidate_L4_Verified,
    "TransportationCandidate_L5_Finalized": TransportationCandidate_L5_Finalized,
    "TravelTransportation": TravelTransportation,
    
    # Accommodation upgrade chain
    "AccommodationCandidate_Raw": AccommodationCandidate_Raw,
    "AccommodationCandidate_L1_Available": AccommodationCandidate_L1_Available,
    "AccommodationCandidate_L2_Located": AccommodationCandidate_L2_Located,
    "AccommodationCandidate_L3_Reviewed": AccommodationCandidate_L3_Reviewed,
    "AccommodationCandidate_L4_Verified": AccommodationCandidate_L4_Verified,
    "AccommodationCandidate_L5_Finalized": AccommodationCandidate_L5_Finalized,
    "TravelAccommodation": TravelAccommodation,
    
    # Attraction upgrade chain
    "AttractionCandidate_Raw": AttractionCandidate_Raw,
    "AttractionCandidate_L1_Available": AttractionCandidate_L1_Available,
    "AttractionCandidate_L2_Located": AttractionCandidate_L2_Located,
    "AttractionCandidate_L3_Reviewed": AttractionCandidate_L3_Reviewed,
    "AttractionCandidate_L4_Verified": AttractionCandidate_L4_Verified,
    "AttractionCandidate_L5_Finalized": AttractionCandidate_L5_Finalized,
    "TravelAttraction": TravelAttraction,
    
    # Dining upgrade chain
    "DiningCandidate_Raw": DiningCandidate_Raw,
    "DiningCandidate_L1_Available": DiningCandidate_L1_Available,
    "DiningCandidate_L2_Located": DiningCandidate_L2_Located,
    "DiningCandidate_L3_Reviewed": DiningCandidate_L3_Reviewed,
    "DiningCandidate_L4_Verified": DiningCandidate_L4_Verified,
    "DiningCandidate_L5_Finalized": DiningCandidate_L5_Finalized,
    "TravelDining": TravelDining,
    
    # Shopping upgrade chain
    "ShoppingCandidate_Raw": ShoppingCandidate_Raw,
    "ShoppingCandidate_L1_Available": ShoppingCandidate_L1_Available,
    "ShoppingCandidate_L2_Located": ShoppingCandidate_L2_Located,
    "ShoppingCandidate_L3_Reviewed": ShoppingCandidate_L3_Reviewed,
    "ShoppingCandidate_L4_Verified": ShoppingCandidate_L4_Verified,
    "ShoppingCandidate_L5_Finalized": ShoppingCandidate_L5_Finalized,
    "TravelShopping": TravelShopping,
    
    # Auxiliary types
    "TimeInfo": TimeInfo,
    "HomeLocation": HomeLocation,
}


def get_data_type_class(type_name: str) -> type:
    """Retrieve the class associated with a type name"""
    if type_name not in DATA_TYPE_REGISTRY:
        raise ValueError(f"Unknown data type: {type_name}")
    return DATA_TYPE_REGISTRY[type_name]


def create_data_object_from_dict(type_name: str, data: Dict[str, Any]) -> DataType:
    """Create a data object from a dictionary"""
    data_type_class = get_data_type_class(type_name)
    return data_type_class.from_dict(data)


def get_preference_types() -> List[str]:
    """Return the names of all preference types"""
    return [
        "LocationPreference",
        "TransportationPreference", 
        "AccommodationPreference",
        "AttractionPreference",
        "DiningPreference",
        "ShoppingPreference"
    ]


def get_candidate_upgrade_chain(subtask: str) -> List[str]:
    """Obtain the candidate upgrade chain for the specified subtask"""
    chains = {
        "location": [
            "LocationCandidate_Raw",
            "LocationCandidate_L1_Available", 
            "LocationCandidate_L2_Located",
            "LocationCandidate_L3_Reviewed",
            "LocationCandidate_L4_Verified",
            "LocationCandidate_L5_Finalized",
            "TravelLocation"
        ],
        "transportation": [
            "TransportationCandidate_Raw",
            "TransportationCandidate_L1_Available",
            "TransportationCandidate_L2_Located",
            "TransportationCandidate_L3_Reviewed",
            "TransportationCandidate_L4_Verified",
            "TransportationCandidate_L5_Finalized",
            "TravelTransportation"
        ],
        "accommodation": [
            "AccommodationCandidate_Raw",
            "AccommodationCandidate_L1_Available",
            "AccommodationCandidate_L2_Located",
            "AccommodationCandidate_L3_Reviewed",
            "AccommodationCandidate_L4_Verified",
            "AccommodationCandidate_L5_Finalized",
            "TravelAccommodation"
        ],
        "attraction": [
            "AttractionCandidate_Raw",
            "AttractionCandidate_L1_Available",
            "AttractionCandidate_L2_Located",
            "AttractionCandidate_L3_Reviewed",
            "AttractionCandidate_L4_Verified",
            "AttractionCandidate_L5_Finalized",
            "TravelAttraction"
        ],
        "dining": [
            "DiningCandidate_Raw",
            "DiningCandidate_L1_Available",
            "DiningCandidate_L2_Located",
            "DiningCandidate_L3_Reviewed",
            "DiningCandidate_L4_Verified",
            "DiningCandidate_L5_Finalized",
            "TravelDining"
        ],
        "shopping": [
            "ShoppingCandidate_Raw",
            "ShoppingCandidate_L1_Available",
            "ShoppingCandidate_L2_Located",
            "ShoppingCandidate_L3_Reviewed",
            "ShoppingCandidate_L4_Verified",
            "ShoppingCandidate_L5_Finalized",
            "TravelShopping"
        ]
    }
    return chains.get(subtask.lower(), [])


def get_final_type(subtask: str) -> str:
    """Get the finalized type for the specified subtask"""
    final_types = {
        "location": "TravelLocation",
        "transportation": "TravelTransportation",
        "accommodation": "TravelAccommodation",
        "attraction": "TravelAttraction",
        "dining": "TravelDining",
        "shopping": "TravelShopping"
    }
    return final_types.get(subtask.lower(), "")


def get_all_subtasks() -> List[str]:
    """Return the names of all subtasks"""
    return ["location", "transportation", "accommodation", "attraction", "dining", "shopping"]

def get_all_goal_type(subtask: str) -> Dict[str, Any]:
    
    goal_type_mapping = {
        "location": "TravelLocation",
        "transportation": "TravelTransportation",
        "accommodation": "TravelAccommodation",
        "attraction": "TravelAttraction",
        "dining": "TravelDining",
        "shopping": "TravelShopping"
    }
    
    return goal_type_mapping.get(subtask.lower(), "")

# Sample usage and test code
if __name__ == "__main__":
    print("=== Data Types Example ===")
    
    # Create a preference
    location_pref = LocationPreference.create(
        category="city",
        tier="bustling_metropolis", 
        style="culture",
        feature_package="history_buff"
    )
    
    # Build the upgrade chain
    raw_candidate = LocationCandidate_Raw.create_from_preference(location_pref)
    l1_candidate = LocationCandidate_L1_Available.create_from_previous(raw_candidate)
    l2_candidate = LocationCandidate_L2_Located.create_from_previous(l1_candidate)
    final_location = TravelLocation.create_from_previous(l2_candidate)
    
    print(f"Raw: {raw_candidate}")
    print(f"L1: {l1_candidate}")
    print(f"L2: {l2_candidate}")
    print(f"Final: {final_location}")
    
    # Test helper functions
    print(f"\nAll subtasks: {get_all_subtasks()}")
    print(f"Location chain: {get_candidate_upgrade_chain('location')}")
    print(f"Transportation final: {get_final_type('transportation')}")
    
""" 
Example testing:
python .\env\core\data_types.py
"""