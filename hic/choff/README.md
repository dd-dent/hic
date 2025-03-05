# CHOFF Parser Module

This module provides a comprehensive implementation of the Cognitive Hoffman Compression Framework (CHOFF) 2.4-RC3 specification.

## Overview

The CHOFF parser module enables parsing and representation of CHOFF expressions in various formats:

- Basic: `{state:type}`
- Intensity-based: `{state:intensity|type1[intensity1]|type2[intensity2]|...}`
- Weighted/Proportional: `{state:weighted|type1[weight1]|type2[weight2]|...}`
- Shorthand for equally-weighted: `{state:weighted:type1|type2[weight]|...}`
- Distribution: `{state:random!type1[weight]!type2[weight]!}`

It also supports context definitions and pattern recognition markers:

- Context: `[context:type]`
- Dynamic Pattern: `&pattern:TYPE|FLOW|`
- Static Status: `&status:TYPE|`

## Usage

### Parsing State Expressions

```python
from hic.choff.parser import parse_state_expression, StateType

# Parse a basic state expression
state = parse_state_expression("{state:analytical}")
print(state.expression_type)  # StateType.BASIC
print(state.state_type)       # "analytical"
print(state.weight)           # 1.0

# Parse an intensity-based state expression
state = parse_state_expression("{state:intensity|analytical[0.8]|creative[0.5]|}")
print(state.expression_type)  # StateType.INTENSITY
print(state.components)       # [StateComponent(state_type="analytical", value=0.8), 
                              #  StateComponent(state_type="creative", value=0.5)]

# Parse a weighted state expression
state = parse_state_expression("{state:weighted|analytical[0.6]|intuitive[0.4]|}")
print(state.expression_type)  # StateType.WEIGHTED
print(state.components)       # [StateComponent(state_type="analytical", value=0.6), 
                              #  StateComponent(state_type="intuitive", value=0.4)]
```

### Parsing Context Definitions

```python
from hic.choff.parser import parse_context

context = parse_context("[context:technical]")
print(context.context_type)  # "technical"
```

### Parsing Pattern Recognition Markers

```python
from hic.choff.parser import parse_pattern

pattern = parse_pattern("&pattern:resonance|active|")
print(pattern.pattern_type)  # "resonance"
print(pattern.flow)         # "active"
print(pattern.is_status)    # False

status = parse_pattern("&status:processing|")
print(status.pattern_type)  # "processing"
print(status.flow)          # None
print(status.is_status)     # True
```

## Integration with Event System

The CHOFF parser integrates with the event system through the enhanced `StateEvent` class, which now supports all CHOFF formats:

```python
from hic.events.schema import StateEvent
from hic.choff.parser import parse_state_expression

# Create a StateEvent from a ChoffState
choff_state = parse_state_expression("{state:intensity|analytical[0.8]|creative[0.5]|}")
event = StateEvent.from_choff_state(
    choff_state=choff_state,
    conversation_id="test-conversation",
    context="problem_solving"
)

# Access properties
print(event.expression_type)  # "intensity"
print(event.state_expression)  # {"analytical": 0.8, "creative": 0.5}

# Backward compatibility
print(event.state_type)  # "analytical"
print(event.intensity)   # 0.8
```

## Backward Compatibility

The module maintains backward compatibility with existing code through:

1. Compatibility properties on `ChoffState` class:
   - `state_type` - Returns the primary state type
   - `weight` - Returns the primary weight/intensity

2. Compatibility properties on `StateEvent` class:
   - `state_type` - Returns the primary state type
   - `intensity` - Returns the primary weight/intensity

3. Support for both string and dictionary state expressions in `StateEvent`.

## Error Handling

The parser provides clear error messages for invalid expressions:

- Invalid format
- Missing components
- Out-of-range values (weights/intensities must be between 0.0 and 1.0)
- Negative values
- Weighted states where weights don't sum to 1.0