
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
    """'''
$html_context
'''
Based on the HTML webpage above, try to complete the following task:
Task: $task
Previous actions:
$previous_actions
What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):   

$choices
"""
)

re_eval_with_html_context_prompt_template = textwrap.dedent("""
You are a visual web assistant. Analyze the attached Screenshot and HTML candidates to select the correct option for the Task.
'''
$html_context
'''
Based on the HTML webpage above, try to complete the following task:
Task: $task
Previous actions:
$previous_actions
What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):   

$choices""")





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

zero_shot_prompt_template = textwrap.dedent(
"""### SYSTEM INSTRUCTION
You are an advanced Multimodal Web Navigation Agent.
Your goal is to select the correct element from the candidates to fulfill the User Task.

**Context:**
- **Screenshot:** The current visual state of the page.
- **HTML:** The DOM elements corresponding to the candidates.

**Process:**
1. **Analyze:** Look at the screenshot to find the element described in the task.
2. **Correlate:** Match that visual element to the correct HTML candidate.
3. **Eliminate:** Explicitly state why the *other* plausible candidates are incorrect.
4. **Decide:** Choose the final correct Option Letter and Action.

**Output Schema Rules:**
- **Option A (None):** If the correct element is NOT listed, output "Answer: A."
- **CLICK:** If the action is clicking a link/button.
  Format: "Answer: [Letter].\nAction: CLICK"
- **HOVER:** If the action is hovering over an element to reveal a menu/tooltip.
  Format: "Answer: [Letter].\nAction: HOVER"
- **TYPE:** If the action is entering text.
  Format: "Answer: [Letter].\nAction: TYPE\nValue: [Exact text to type]"
- **SELECT:** If the action is choosing from a dropdown.
  Format: "Answer: [Letter].\nAction: SELECT\nValue: [Option text to select]"

**Output Format:**
- **Step 1:** Write your analysis inside a `[[THOUGHT]]` block.
- **Step 2:** Write the strict Mind2Web format immediately after inside an `[[ANSWER]]` block.

**Example Output:**
[[THOUGHT]]
The user wants to open the "Services" submenu. In the screenshot, "Services" is a main nav item that shows a dropdown when hovered.
- Option B is the "Services" link.
- Clicking might go to a different page, but the task implies navigating the menu structure first.
- Therefore, the correct action is to hover.
[[ANSWER]]
Answer: B.
Action: HOVER

### TASK CONTEXT
**Task:** $task
**Previous Actions:**
$prev_actions

### HTML CONTEXT
$html_context

### CANDIDATE OPTIONS
$choices

### FINAL REMINDER
You MUST end your response with the [[ANSWER]] block containing the strict action format. Do not stop after the thoughts.

### YOUR RESPONSE
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
    if name == "re_eval_with_context": return Template(re_eval_with_html_context_prompt_template)
    if name == "zero_shot":
        return Template(zero_shot_prompt_template)
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
