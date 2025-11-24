
import textwrap

# Templates uses $ instead of {}
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


