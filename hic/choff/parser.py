"""CHOFF parsing module for state expressions, context definitions, and pattern recognition."""
import re
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field


class StateType(Enum):
    """Types of CHOFF state expressions."""
    INTENSITY = "intensity"
    WEIGHTED = "weighted"
    RANDOM = "random"
    BASIC = "basic"


@dataclass
class StateComponent:
    """Individual state component with weight/intensity."""
    state_type: str
    value: float


@dataclass
class ChoffState:
    """Enhanced CHOFF state representation."""
    expression_type: StateType
    components: List[StateComponent]
    
    @property
    def state_type(self) -> str:
        """Get the primary state type for backward compatibility."""
        if not self.components:
            return ""
        return self.components[0].state_type
    
    @property
    def weight(self) -> float:
        """Get the primary weight for backward compatibility."""
        if not self.components:
            return 1.0
        return self.components[0].value


@dataclass
class ChoffContext:
    """CHOFF context representation."""
    context_type: str
    
    @classmethod
    def from_tag(cls, tag: str) -> 'ChoffContext':
        """Parse a CHOFF context tag."""
        return parse_context(tag)


@dataclass
class ChoffPattern:
    """CHOFF pattern representation."""
    pattern_type: str
    flow: Optional[str] = None
    is_status: bool = False
    
    @classmethod
    def from_tag(cls, tag: str) -> 'ChoffPattern':
        """Parse a CHOFF pattern tag."""
        return parse_pattern(tag)


def parse_state_expression(expr: str) -> ChoffState:
    """Parse a CHOFF state expression into type and components.
    
    Supports the following formats:
    - Basic: {state:type}
    - Intensity-based: {state:intensity|type1[intensity1]|type2[intensity2]|...}
    - Weighted/Proportional: {state:weighted|type1[weight1]|type2[weight2]|...}
    - Shorthand for equally-weighted: {state:weighted:type1|type2[weight]|...}
    - Distribution: {state:random!type1[weight]!type2[weight]!}
    
    Args:
        expr: CHOFF state expression string
        
    Returns:
        ChoffState object with expression type and components
        
    Raises:
        ValueError: If the expression is invalid
    """
    if not expr:
        raise ValueError("Empty state expression")
    
    # Basic format: {state:type}
    basic_match = re.match(r'\{state:([^}\[\|:!]+)\}', expr)
    if basic_match:
        state_type = basic_match.group(1).strip()
        if not state_type:
            raise ValueError("State type cannot be empty")
        return ChoffState(
            expression_type=StateType.BASIC,
            components=[StateComponent(state_type=state_type, value=1.0)]
        )
    
    # Basic format with weight: {state:type[weight]}
    basic_weight_match = re.match(r'\{state:([^}\[\|:!]+)\[([-0-9.]+)\]\}', expr)
    if basic_weight_match:
        state_type = basic_weight_match.group(1).strip()
        weight_str = basic_weight_match.group(2)
        try:
            weight = float(weight_str)
            if weight < 0.0 or weight > 1.0:
                raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")
        except ValueError as e:
            raise ValueError(f"Invalid weight: {weight_str}") from e
        
        return ChoffState(
            expression_type=StateType.BASIC,
            components=[StateComponent(state_type=state_type, value=weight)]
        )
    
    # Intensity-based format: {state:intensity|type1[intensity1]|type2[intensity2]|...}
    intensity_match = re.match(r'\{state:intensity\|(.+)\|\}', expr)
    if intensity_match:
        components_str = intensity_match.group(1)
        if not components_str.strip():
            raise ValueError("No components found in intensity-based expression")
            
        components = []
        
        # Parse components
        component_pattern = r'([^|\[\]]+)(?:\[([-0-9.]+)\])?'
        for match in re.finditer(component_pattern, components_str):
            state_type = match.group(1).strip()
            intensity_str = match.group(2)
            
            if not state_type:
                continue
            
            try:
                if intensity_str and intensity_str.startswith('-'):
                    raise ValueError(f"Intensity cannot be negative: {intensity_str}")
                
                intensity = float(intensity_str) if intensity_str else 1.0
                if intensity < 0.0 or intensity > 1.0:
                    raise ValueError(f"Intensity must be between 0.0 and 1.0, got {intensity}")
                components.append(StateComponent(state_type=state_type, value=intensity))
            except ValueError as e:
                raise ValueError(f"Invalid intensity: {intensity_str}") from e
        
        if not components:
            raise ValueError("No valid components found in intensity-based expression")
        
        return ChoffState(
            expression_type=StateType.INTENSITY,
            components=components
        )
    
    # Weighted/Proportional format: {state:weighted|type1[weight1]|type2[weight2]|...}
    weighted_match = re.match(r'\{state:weighted\|(.+)\|\}', expr)
    if weighted_match:
        components_str = weighted_match.group(1)
        if not components_str.strip():
            raise ValueError("No components found in weighted expression")
            
        components = []
        
        # Parse components
        component_pattern = r'([^|\[\]]+)(?:\[([-0-9.]+)\])?'
        for match in re.finditer(component_pattern, components_str):
            state_type = match.group(1).strip()
            weight_str = match.group(2)
            
            if not state_type:
                continue
            
            try:
                weight = float(weight_str) if weight_str else 1.0
                if weight < 0.0 or weight > 1.0:
                    raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")
                components.append(StateComponent(state_type=state_type, value=weight))
            except ValueError as e:
                raise ValueError(f"Invalid weight: {weight_str}") from e
        
        if not components:
            raise ValueError("No valid components found in weighted expression")
        
        # Validate weights sum to 1.0
        total_weight = sum(component.value for component in components)
        if not pytest.approx(total_weight) == 1.0:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        return ChoffState(
            expression_type=StateType.WEIGHTED,
            components=components
        )
    
    # Shorthand for equally-weighted: {state:weighted:type1|type2[weight]|...}
    shorthand_match = re.match(r'\{state:weighted:(.+)\|\}', expr)
    if shorthand_match:
        components_str = shorthand_match.group(1)
        components = []
        weighted_components = []
        unweighted_types = []
        
        # Parse components
        component_pattern = r'([^|\[\]]+)(?:\[([-0-9.]+)\])?'
        for match in re.finditer(component_pattern, components_str):
            state_type = match.group(1).strip()
            weight_str = match.group(2)
            
            if not state_type:
                continue
            
            if weight_str:
                try:
                    weight = float(weight_str)
                    if weight < 0.0 or weight > 1.0:
                        raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")
                    weighted_components.append(StateComponent(state_type=state_type, value=weight))
                except ValueError as e:
                    raise ValueError(f"Invalid weight: {weight_str}") from e
            else:
                unweighted_types.append(state_type)
        
        # Calculate weight for unweighted types
        total_weighted = sum(component.value for component in weighted_components)
        if total_weighted > 1.0:
            raise ValueError(f"Weights exceed 1.0: {total_weighted}")
        
        remaining_weight = 1.0 - total_weighted
        if unweighted_types:
            weight_per_type = remaining_weight / len(unweighted_types)
            for state_type in unweighted_types:
                components.append(StateComponent(state_type=state_type, value=weight_per_type))
        
        components.extend(weighted_components)
        
        if not components:
            raise ValueError("No valid components found in shorthand weighted expression")
        
        return ChoffState(
            expression_type=StateType.WEIGHTED,
            components=components
        )
    
    # Distribution format: {state:random!type1[weight]!type2[weight]!}
    random_match = re.match(r'\{state:random!(.+)!\}', expr)
    if random_match:
        components_str = random_match.group(1)
        if not components_str.strip():
            raise ValueError("No components found in random distribution expression")
            
        components = []
        
        # Parse components
        component_pattern = r'([^!\[\]]+)(?:\[([-0-9.]+)\])?'
        for match in re.finditer(component_pattern, components_str):
            state_type = match.group(1).strip()
            weight_str = match.group(2)
            
            if not state_type:
                continue
            
            try:
                weight = float(weight_str) if weight_str else 1.0
                if weight < 0.0 or weight > 1.0:
                    raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")
                components.append(StateComponent(state_type=state_type, value=weight))
            except ValueError as e:
                raise ValueError(f"Invalid weight: {weight_str}") from e
        
        if not components:
            raise ValueError("No valid components found in random distribution expression")
        
        # Validate weights sum to 1.0
        total_weight = sum(component.value for component in components)
        if not pytest.approx(total_weight) == 1.0:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        return ChoffState(
            expression_type=StateType.RANDOM,
            components=components
        )
    
    raise ValueError(f"Invalid state expression format: {expr}")


def parse_context(expr: str) -> ChoffContext:
    """Parse a CHOFF context expression.
    
    Format: [context:type]
    
    Args:
        expr: CHOFF context expression string
        
    Returns:
        ChoffContext object
        
    Raises:
        ValueError: If the expression is invalid
    """
    if not expr:
        raise ValueError("Empty context expression")
    
    match = re.match(r'\[context:([^\]]+)\]', expr)
    if not match:
        raise ValueError(f"Invalid CHOFF context tag: {expr}")
    
    context_type = match.group(1).strip()
    if not context_type:
        raise ValueError("Context type cannot be empty")
    
    return ChoffContext(context_type=context_type)


def parse_pattern(expr: str) -> ChoffPattern:
    """Parse a CHOFF pattern expression.
    
    Formats:
    - Dynamic Pattern: &pattern:TYPE|FLOW|
    - Static Status: &status:TYPE|
    
    Args:
        expr: CHOFF pattern expression string
        
    Returns:
        ChoffPattern object
        
    Raises:
        ValueError: If the expression is invalid
    """
    if not expr:
        raise ValueError("Empty pattern expression")
    
    # Dynamic Pattern: &pattern:TYPE|FLOW|
    pattern_match = re.match(r'&pattern:([^|]+)\|([^|]+)\|', expr)
    if pattern_match:
        pattern_type = pattern_match.group(1).strip()
        flow = pattern_match.group(2).strip()
        
        if not pattern_type:
            raise ValueError("Pattern type cannot be empty")
        if not flow:
            raise ValueError("Flow cannot be empty")
        
        return ChoffPattern(
            pattern_type=pattern_type,
            flow=flow,
            is_status=False
        )
    
    # Static Status: &status:TYPE|
    status_match = re.match(r'&status:([^|]+)\|', expr)
    if status_match:
        status_type = status_match.group(1).strip()
        
        if not status_type:
            raise ValueError("Status type cannot be empty")
        
        return ChoffPattern(
            pattern_type=status_type,
            flow=None,
            is_status=True
        )
    
    raise ValueError(f"Invalid CHOFF pattern tag: {expr}")


# Add pytest for approx function used in validation
import pytest