
import textwrap
from string import Template

standard_prompt_template = textwrap.dedent(
"""
$html
Based on the HTML webpage above, try to complete the following task:
Task: $task
Previous actions:
$prev_actions

What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):

$choices
"""
)


mind2web_prompt_template = textwrap.dedent(
    """
Based on the HTML webpage above, try to complete the following task:
Task: $task
Previous actions:
$prev_actions
What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):

$choices
"""
)


mind2web_prompt_template_with_html_context= textwrap.dedent(
    """
    '''
$html
    '''

Based on the HTML webpage above, try to complete the following task:
Task: $task
Previous actions:
$prev_actions
What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):

$choices
"""
)



# Templates uses $ instead of {}
# Change to human orale
oracle_prompt_template = textwrap.dedent(
    """
    ### ROLE
    You are a helpful human annotator assisting a web navigation agent. The agent is confused and has narrowed down the next step to a few likely options, but it cannot distinguish which one is correct.
    
    ### USER TASK
    $task
    Previous Actions: 
    $previous_actions

    ### HTML CONTEXT
    $html

    ### AVAILABLE OPTIONS
    The smaller model has provided the following possible actions:
    $options

    ### THE AMBIGUOUS OPTIONS (Prediction Set)
    The correct action is very likely one of the following. Analyze them carefully:
    $prediction_set_options

    ### INSTRUCTIONS
    1. Analyze the User Task and the HTML Context (and the screenshot if available).
    2. Identify the single correct element from the Prediction Set options above.
    3. Formulate a natural language instruction to guide the agent to that specific element.
    4. **Crucial:** Your instruction must be "Discriminative". Mention specific text labels, icons, or the location of the element to help the agent separate it from the other options.
  
    ### RESPONSE FORMAT
    Provide your response as a direct command to the agent. 
    - If the action is clicking, describe exactly what to click.
    - If the action is typing, specify exactly what to type and where.

    Example 1 (Click): "Click the 'Menu' button found in the top-left navigation bar."
    Example 2 (Type): "Type 'The Matrix' into the search bar located in the header."
    Example 3 (Select): "Select 'Price: Low to High' from the dropdown menu."

    Your Response:  
""")

re_eval_prompt_template = textwrap.dedent(
    """Based on the HTML webpage above, try to complete the following task:
Task: $task
Guidence: $help
Previous actions:
$prev_actions
What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):
$choices
"""
)

human_prompt_template = textwrap.dedent(
    """
    ### ROLE
    You are a helpful human annotator assisting a web navigation agent. The agent is confused and has narrowed down the next step to a few likely options, but it cannot distinguish which one is correct.
    ### USER TASK
    $task
    Previous Actions: 
    $previous_actions

    ### AVAILABLE OPTIONS
    The smaller model has provided the following possible actions:
    $options

    ### THE AMBIGUOUS OPTIONS (Prediction Set)
    The correct action is very likely one of the following. Analyze them carefully:
    $prediction_set_options

    ### INSTRUCTIONS
    1. Analyze the User Task and the HTML Context (and the screenshot if available).
    2. Identify the single correct element from the Prediction Set options above.
    3. Formulate a natural language instruction to guide the agent to that specific element.
    4. **Crucial:** Your instruction must be "Discriminative". Mention specific text labels, icons, or the location of the element to help the agent separate it from the other options.
  
    ### RESPONSE FORMAT
    Provide your response as a direct command to the agent. 
    - If the action is clicking, describe exactly what to click.
    - If the action is typing, specify exactly what to type and where.

    Example 1 (Click): "Click the 'Menu' button found in the top-left navigation bar."
    Example 2 (Type): "Type 'The Matrix' into the search bar located in the header."
    Example 3 (Select): "Select 'Price: Low to High' from the dropdown menu."

    Your Response:  
""")


def get_prompt_template(name: str):
    """
    Return a Template object for the given prompt name.

    Valid names: "standard", "mind2web", "oracle", "re_eval", "human".
    """
    name = name.lower()
    if name == "standard":
        return Template(standard_prompt_template)
    if name == "mind2web":
        return Template(mind2web_prompt_template)
    if name == "mind2web_with_context":
        return Template(mind2web_prompt_template_with_html_context)
    if name == "oracle":
        return Template(oracle_prompt_template)
    if name == "re_eval":
        return Template(re_eval_prompt_template)
    if name == "human":
        return Template(human_prompt_template)
    raise ValueError(f"Unknown prompt template name: {name}")



# """

# Task: Find the 'Submit' button.
# Choices:
# A. None of the above
# B. <button id=0> Cancel
# C. <button id=1> Submit
# D. <div id=2> Footer

# Think step by step. First "Reasoning:", then "Answer:".


# """
