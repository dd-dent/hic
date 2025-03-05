"""Tests for the enhanced StateEvent class."""
import pytest
from typing import Dict, List, Optional

from hic.events.schema import StateEvent, validate_event
from hic.choff.parser import (
    parse_state_expression,
    StateType,
    ChoffState,
    StateComponent
)

class TestStateEvent:
    """Tests for the enhanced StateEvent class."""

    def test_create_with_string_expression(self):
        """Test creating a StateEvent with a string expression."""
        event = StateEvent.create(
            state_expression="analytical",
            conversation_id="test-conversation",
            expression_type="basic",
            context="technical"
        )
        
        # Verify event properties
        assert event.state_expression == "analytical"
        assert event.expression_type == "basic"
        assert event.context == "technical"
        
        # Verify backward compatibility
        assert event.state_type == "analytical"
        assert event.intensity == 1.0
        
        # Validate event
        validate_event(event)
    
    def test_create_with_dict_expression(self):
        """Test creating a StateEvent with a dictionary expression."""
        state_expression = {
            "analytical": 0.8,
            "creative": 0.5
        }
        
        event = StateEvent.create(
            state_expression=state_expression,
            conversation_id="test-conversation",
            expression_type="intensity",
            context="problem_solving"
        )
        
        # Verify event properties
        assert event.state_expression == state_expression
        assert event.expression_type == "intensity"
        assert event.context == "problem_solving"
        
        # Verify backward compatibility (using first item)
        assert event.state_type == "analytical"
        assert event.intensity == 0.8
        
        # Validate event
        validate_event(event)
    
    def test_from_choff_state_basic(self):
        """Test creating a StateEvent from a basic ChoffState."""
        # Create a ChoffState
        choff_state = parse_state_expression("{state:analytical}")
        
        # Create StateEvent from ChoffState
        event = StateEvent.from_choff_state(
            choff_state=choff_state,
            conversation_id="test-conversation",
            context="technical"
        )
        
        # Verify event properties
        assert event.state_expression == "analytical"
        assert event.expression_type == "basic"
        assert event.context == "technical"
        
        # Validate event
        validate_event(event)
    
    def test_from_choff_state_intensity(self):
        """Test creating a StateEvent from an intensity-based ChoffState."""
        # Create a ChoffState
        choff_state = parse_state_expression("{state:intensity|analytical[0.8]|creative[0.5]|}")
        
        # Create StateEvent from ChoffState
        event = StateEvent.from_choff_state(
            choff_state=choff_state,
            conversation_id="test-conversation",
            context="problem_solving"
        )
        
        # Verify event properties
        assert isinstance(event.state_expression, dict)
        assert event.state_expression["analytical"] == 0.8
        assert event.state_expression["creative"] == 0.5
        assert event.expression_type == "intensity"
        assert event.context == "problem_solving"
        
        # Validate event
        validate_event(event)
    
    def test_from_choff_state_weighted(self):
        """Test creating a StateEvent from a weighted ChoffState."""
        # Create a ChoffState
        choff_state = parse_state_expression("{state:weighted|analytical[0.6]|intuitive[0.4]|}")
        
        # Create StateEvent from ChoffState
        event = StateEvent.from_choff_state(
            choff_state=choff_state,
            conversation_id="test-conversation",
            context="decision_making"
        )
        
        # Verify event properties
        assert isinstance(event.state_expression, dict)
        assert event.state_expression["analytical"] == 0.6
        assert event.state_expression["intuitive"] == 0.4
        assert event.expression_type == "weighted"
        assert event.context == "decision_making"
        
        # Validate event
        validate_event(event)
    
    def test_validation_errors(self):
        """Test validation errors for StateEvent."""
        # Test missing state expression
        with pytest.raises(ValueError, match="State expression is required"):
            event = StateEvent(
                metadata=StateEvent.create("test", "test-conversation").metadata,
                state_expression="",
                expression_type="basic"
            )
            validate_event(event)
        
        # Test invalid state value
        with pytest.raises(ValueError, match="State value must be between 0.0 and 1.0"):
            event = StateEvent(
                metadata=StateEvent.create("test", "test-conversation").metadata,
                state_expression={"analytical": 1.1},
                expression_type="intensity"
            )
            validate_event(event)