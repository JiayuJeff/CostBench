from enum import Enum
from typing import List, Dict, Any


class LocationCategory(Enum):
    CITY = "city"
    SEASIDE = "seaside"
    MOUNTAIN = "mountain"
    VILLAGE = "village"


class LocationTier(Enum):
    MAJOR_METROPOLIS = "major_metropolis"
    MID_SIZED_CITY = "mid_sized_city"
    SMALL_TOWN = "small_town"
    SECLUDED_AREA = "secluded_area"


class LocationStyle(Enum):
    HISTORIC_AND_TRADITIONAL = "historic_and_traditional"
    MODERN_AND_COSMOPOLITAN = "modern_and_cosmopolitan"
    NATURAL_AND_SERENE = "natural_and_serene"
    ENTERTAINMENT_AND_VIBRANT = "entertainment_and_vibrant"


class LocationFeaturePackage(Enum):
    ARCHITECTURAL_MARVEL = "architectural_marvel"
    RELIGIOUS_CENTER = "religious_center"
    SIGNATURE_THEME_CITY = "signature_theme_city"
    CULINARY_CAPITAL = "culinary_capital"


class TransportationCategory(Enum):
    FLIGHT = "flight"
    TRAIN = "train"
    BUS = "bus"
    CAR_RENTAL = "car_rental"


class TransportationTier(Enum):
    LUXURY_CLASS = "luxury_class"
    BUSINESS_CLASS = "business_class"
    STANDARD_CLASS = "standard_class"
    BUDGET_CLASS = "budget_class"


class TransportationStyle(Enum):
    SPEED_PRIORITY = "speed_priority"
    COMFORT_PRIORITY = "comfort_priority"
    SCENIC_ROUTE = "scenic_route"
    SCHEDULE_FLEXIBILITY_PRIORITY = "schedule_flexibility_priority"


class TransportationFeaturePackage(Enum):
    ONBOARD_CONNECTIVITY_AND_POWER = "onboard_connectivity_and_power"
    FULL_MEAL_AND_BEVERAGE_SERVICE = "full_meal_and_beverage_service"
    SPECIAL_LUGGAGE_ALLOWANCE = "special_luggage_allowance"
    LIE_FLAT_OR_SLEEPER_FACILITY = "lie_flat_or_sleeper_facility"


class AccommodationCategory(Enum):
    HOTEL = "hotel"
    CAMPSITE = "campsite"
    HOMESTAY = "homestay"
    HOSTEL = "hostel"


class AccommodationTier(Enum):
    LUXURY_FIVE_STAR = "luxury_five_star"
    BUSINESS_FOUR_STAR = "business_four_star"
    STANDARD_THREE_STAR = "standard_three_star"
    BUDGET_BASIC = "budget_basic"


class AccommodationStyle(Enum):
    CENTRALITY_PRIORITY = "centrality_priority"
    SCENERY_PRIORITY = "scenery_priority"
    CONVENIENCE_PRIORITY = "convenience_priority"
    TRANQUILITY_PRIORITY = "tranquility_priority"


class AccommodationFeaturePackage(Enum):
    RESORT_STYLE_RECREATION = "resort_style_recreation"
    SPECIALIZED_PET_FRIENDLY_SERVICES = "specialized_pet_friendly_services"
    COMPREHENSIVE_BUSINESS_SERVICES = "comprehensive_business_services"
    SELF_CATERING_AND_LAUNDRY = "self_catering_and_laundry"


class AttractionCategory(Enum):
    HISTORIC_SITE = "historic_site"
    MUSEUM_OR_GALLERY = "museum_or_gallery"
    NATURAL_WONDER = "natural_wonder"
    AMUSEMENT_OR_ENTERTAINMENT = "amusement_or_entertainment"


class AttractionTier(Enum):
    GLOBAL_LANDMARK = "global_landmark"
    NATIONAL_TREASURE = "national_treasure"
    REGIONAL_HIGHLIGHT = "regional_highlight"
    LOCAL_SPOT = "local_spot"


class AttractionStyle(Enum):
    LEARNING_AND_KNOWLEDGE_PRIORITY = "learning_and_knowledge_priority"
    ACTION_AND_THRILL_PRIORITY = "action_and_thrill_priority"
    RELAXATION_AND_AESTHETICS_PRIORITY = "relaxation_and_aesthetics_priority"
    SOCIAL_AND_COMMUNITY_PRIORITY = "social_and_community_priority"


class AttractionFeaturePackage(Enum):
    IN_DEPTH_EXPERT_GUIDANCE = "in_depth_expert_guidance"
    LIVE_PERFORMANCES = "live_performances"
    HANDS_ON_CREATION_EXPERIENCE = "hands_on_creation_experience"
    IMMERSIVE_THEMED_DINING = "immersive_themed_dining"


class DiningCategory(Enum):
    COFFEE_SHOP = "coffee_shop"
    RESTAURANT = "restaurant"
    STREET_FOOD_VENDOR = "street_food_vendor"
    BAKERY = "bakery"


class DiningTier(Enum):
    WORLD_RENOWNED_AWARD_WINNER = "world_renowned_award_winner"
    NATIONALLY_ACCLAIMED = "nationally_acclaimed"
    POPULAR_LOCAL_MAINSTAY = "popular_local_mainstay"
    SIMPLE_AND_BUDGET_EATS = "simple_and_budget_eats"


class DiningStyle(Enum):
    CULINARY_QUALITY_PRIORITY = "culinary_quality_priority"
    AMBIANCE_AND_SETTING_PRIORITY = "ambiance_and_setting_priority"
    SPEED_AND_CONVENIENCE_PRIORITY = "speed_and_convenience_priority"
    VALUE_FOR_MONEY_PRIORITY = "value_for_money_priority"


class DiningFeaturePackage(Enum):
    LIVE_MUSIC_OR_PERFORMANCE = "live_music_or_performance"
    EXCEPTIONAL_VIEW_OR_SETTING = "exceptional_view_or_setting"
    CHEF_INTERACTION_AND_OPEN_KITCHEN = "chef_interaction_and_open_kitchen"
    UNIQUE_THEMED_ENVIRONMENT = "unique_themed_environment"


class ShoppingCategory(Enum):
    SHOPPING_CENTER_OR_MALL = "shopping_center_or_mall"
    SHOPPING_STREET = "shopping_street"
    LOCAL_MARKET_OR_BAZAAR = "local_market_or_bazaar"
    BOUTIQUES_AND_SPECIALTY_STORES = "boutiques_and_specialty_stores"


class ShoppingTier(Enum):
    HIGH_END_LUXURY = "high_end_luxury"
    MID_RANGE_BRAND = "mid_range_brand"
    AFFORDABLE_PRICE = "affordable_price"
    BARGAIN_DEALS = "bargain_deals"


class ShoppingStyle(Enum):
    BRAND_AND_QUALITY_PRIORITY = "brand_and_quality_priority"
    PRICE_PRIORITY = "price_priority"
    UNIQUENESS_AND_DISCOVERY_PRIORITY = "uniqueness_and_discovery_priority"
    EFFICIENCY_AND_NEEDS_PRIORITY = "efficiency_and_needs_priority"


class ShoppingFeaturePackage(Enum):
    DUTY_FREE = "duty_free"
    PERSONAL_SHOPPING_OR_STYLING_SERVICE = "personal_shopping_or_styling_service"
    LOCAL_ARTISAN_AND_HANDCRAFT_SPECIALTY = "local_artisan_and_handcraft_specialty"
    VINTAGE_AND_SECOND_HAND_SPECIALTY = "vintage_and_second_hand_specialty"
    
    
ENUM_DIRECT_MAPPINGS = {
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
    "ShoppingFeaturePackage": ShoppingFeaturePackage
}
    
    
ENUM_MAPPINGS = {
    "location": {
        "search": [
            LocationCategory,
            LocationTier,
            LocationStyle,
            LocationFeaturePackage
        ],
    },
    "transportation": {
        "search": [
            TransportationCategory,
            TransportationTier,
            TransportationStyle,
            TransportationFeaturePackage
        ]
    },
    "accommodation": {
        "search": [
            AccommodationCategory,
            AccommodationTier,
            AccommodationStyle,
            AccommodationFeaturePackage
        ]
    },
    "attraction": {
        "search": [
            AttractionCategory,
            AttractionTier,
            AttractionStyle,
            AttractionFeaturePackage
        ]
    },
    "dining": {
        "search": [
            DiningCategory,
            DiningTier,
            DiningStyle,
            DiningFeaturePackage
        ]
    },
    "shopping": {
        "search": [
            ShoppingCategory,
            ShoppingTier,
            ShoppingStyle,
            ShoppingFeaturePackage
        ]
    }
}