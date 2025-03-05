"""Tests for the CHOFF parser module."""
import pytest
from hypothesis import given, strategies as st
from typing import Dict, List, Tuple, Optional

from hic.choff.parser import (
    parse_state_expression,
    parse_context,
    parse_pattern,
    StateType,
    StateComponent,
    ChoffState,
    ChoffContext,
    ChoffPattern
)

class TestStateExpressionParser:
    """Tests for the state expression parser."""

    @pytest.mark.parametrize("input_str, expected_type, expected_components", [
        # Basic format
        ("{state:analytical}", StateType.BASIC, {"analytical": 1.0}),
        ("{state:creative}", StateType.BASIC, {"creative": 1.0}),
        
        # Intensity-based format
        ("{state:intensity|analytical[0.8]|creative[0.5]|}", StateType.INTENSITY, 
         {"analytical": 0.8, "creative": 0.5}),
        ("{state:intensity|focused[1.0]|}", StateType.INTENSITY, {"focused": 1.0}),
        
        # Weighted/Proportional format
        ("{state:weighted|analytical[0.6]|intuitive[0.4]|}", StateType.WEIGHTED, 
         {"analytical": 0.6, "intuitive": 0.4}),
        ("{state:weighted|methodical[0.7]|creative[0.3]|}", StateType.WEIGHTED, 
         {"methodical": 0.7, "creative": 0.3}),
        
        # Shorthand for equally-weighted
        ("{state:weighted:reflective|analytical[0.5]|}", StateType.WEIGHTED, 
         {"reflective": 0.5, "analytical": 0.5}),
        ("{state:weighted:a|b|c|}", StateType.WEIGHTED, 
         {"a": 1/3, "b": 1/3, "c": 1/3}),
        
        # Distribution format
        ("{state:random!optimistic[0.5]!skeptical[0.5]!}", StateType.RANDOM, 
         {"optimistic": 0.5, "skeptical": 0.5}),
    ])
    def test_state_expression_parsing(self, input_str, expected_type, expected_components):
        """Test parsing of various state expression formats."""
        result = parse_state_expression(input_str)
        assert result.expression_type == expected_type
        
        # Check components match expected values
        assert len(result.components) == len(expected_components)
        for component in result.components:
            assert component.state_type in expected_components
            assert pytest.approx(component.value) == expected_components[component.state_type]
    
    def test_invalid_state_expressions(self):
        """Test parsing of invalid state expressions."""
        # Test empty string
        with pytest.raises(ValueError):
            parse_state_expression("")
            
        # Test missing braces
        with pytest.raises(ValueError):
            parse_state_expression("state:analytical")
            
        # Test missing type
        with pytest.raises(ValueError):
            parse_state_expression("{state:}")
            
        # Test no components in weighted
        with pytest.raises(ValueError):
            parse_state_expression("{state:weighted|}")
            
        # Test value out of range
        with pytest.raises(ValueError):
            parse_state_expression("{state:weighted|analytical[1.1]|}")
            
        # Test wrong delimiter
        with pytest.raises(ValueError):
            parse_state_expression("{state:random!optimistic[0.5]|}")
            
        # Test negative value
        with pytest.raises(ValueError):
            parse_state_expression("{state:intensity|analytical[-0.1]|}")
    
    @given(st.text())
    def test_parser_robustness(self, s):
        """Test that parser handles any input without crashing."""
        try:
            result = parse_state_expression(s)
            # Should either return a valid result or raise a ValueError
        except ValueError:
            # Expected for invalid input
            pass
        # Should never raise any other exception


class TestContextParser:
    """Tests for the context parser."""

    @pytest.mark.parametrize("input_str, expected_type", [
        ("[context:technical]", "technical"),
        ("[context:meta]", "meta"),
        ("[context:problem_solving]", "problem_solving"),
    ])
    def test_context_parsing(self, input_str, expected_type):
        """Test parsing of context expressions."""
        result = parse_context(input_str)
        assert result.context_type == expected_type
    
    def test_invalid_context_expressions(self):
        """Test parsing of invalid context expressions."""
        invalid_expressions = [
            "",  # Empty string
            "context:technical",  # Missing brackets
            "[context:]",  # Missing type
        ]
        
        for expr in invalid_expressions:
            with pytest.raises(ValueError):
                parse_context(expr)


class TestPatternParser:
    """Tests for the pattern parser."""

    @pytest.mark.parametrize("input_str, expected_type, expected_flow, expected_is_status", [
        ("&pattern:resonance|active|", "resonance", "active", False),
        ("&pattern:insight|passive|", "insight", "passive", False),
        ("&status:processing|", "processing", None, True),
    ])
    def test_pattern_parsing(self, input_str, expected_type, expected_flow, expected_is_status):
        """Test parsing of pattern expressions."""
        result = parse_pattern(input_str)
        assert result.pattern_type == expected_type
        assert result.flow == expected_flow
        assert result.is_status == expected_is_status
    
    def test_invalid_pattern_expressions(self):
        """Test parsing of invalid pattern expressions."""
        invalid_expressions = [
            "",  # Empty string
            "pattern:resonance|active|",  # Missing ampersand
            "&pattern:|active|",  # Missing type
            "&pattern:resonance||",  # Missing flow
            "&status:|",  # Missing type
        ]
        
        for expr in invalid_expressions:
            with pytest.raises(ValueError):
                parse_pattern(expr)


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_choff_state_compatibility(self):
        """Test that ChoffState is compatible with existing code."""
        # Create a state using the new parser
        state = parse_state_expression("{state:analytical[0.8]}")
        
        # Verify properties for backward compatibility
        assert state.state_type == "analytical"
        assert state.weight == 0.8

    def test_state_event_compatibility(self):
        """Test that StateEvent can be created from ChoffState."""
        # This test will be implemented after updating StateEvent
        pass