"""
Presets utility module for handling preset configurations across different tools
"""

from typing import Dict, List, Any, Union, Optional
from .config import PRESETS

def get_preset_teams() -> Dict[str, str]:
    """Get all available team presets"""
    return PRESETS["teams"]

def get_preset_players(category: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
    """
    Get player presets, either all categories or a specific category
    
    Args:
        category: Optional category name to retrieve specific player group
    
    Returns:
        Either the full players preset dictionary or a specific list of players
    """
    if category and category in PRESETS["players"]:
        return PRESETS["players"][category]
    return PRESETS["players"]

def get_preset_draft_settings() -> Dict[str, Dict[str, Any]]:
    """Get all available draft setting presets"""
    return PRESETS["draft"]

def get_consistency_presets() -> List[str]:
    """Get preset players for consistency tracking"""
    return PRESETS["consistency"]

def load_team_preset(preset_name: str) -> str:
    """
    Load a specific team preset by name
    
    Args:
        preset_name: The name of the preset to load
        
    Returns:
        String of comma-separated player names
    """
    return PRESETS["teams"].get(preset_name.lower().replace(" ", "_"), "")

def create_preset_buttons(gr, container, preset_dict: Dict[str, Any], 
                         output_component, value_key: Optional[str] = None,
                         css_class: str = "preset-button") -> Dict[str, Any]:
    """
    Create a grid of preset buttons in the provided container
    
    Args:
        gr: Gradio module
        container: Gradio container to place buttons in
        preset_dict: Dictionary of preset values 
        output_component: Component to receive the preset value
        value_key: Optional key to extract from preset dictionary values
        css_class: CSS class to apply to buttons
        
    Returns:
        Dictionary of created button objects
    """
    buttons = {}
    
    # Create a grid for the buttons
    with container:
        for preset_name, preset_value in preset_dict.items():
            # Format the display name
            display_name = preset_name.replace('_', ' ').title()
            
            # Create the button
            btn = gr.Button(display_name, elem_classes=[css_class])
            
            # Determine the value to set
            if value_key and isinstance(preset_value, dict):
                set_value = preset_value.get(value_key)
            else:
                set_value = preset_value
                
            # Set up click handler
            btn.click(
                fn=lambda v=set_value: v,
                inputs=[],
                outputs=[output_component]
            )
            
            buttons[preset_name] = btn
            
    return buttons

def create_preset_section(gr, title: str, container, presets: Dict[str, Any], 
                         output_component, value_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a complete preset section with title and buttons
    
    Args:
        gr: Gradio module
        title: Section title
        container: Gradio container to place section in
        presets: Dictionary of preset values
        output_component: Component to receive preset values
        value_key: Optional key to extract from preset dictionary values
        
    Returns:
        Dictionary of created button objects
    """
    with container:
        with gr.Column(elem_classes=["preset-section"]):
            gr.Markdown(f"#### {title}")
            with gr.Column(elem_classes=["preset-grid"]):
                buttons = create_preset_buttons(
                    gr, gr.Column(), presets, output_component, value_key
                )
    
    return buttons 