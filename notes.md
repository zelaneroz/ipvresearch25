# Knowledge Gained throughout the Process

## Prompting Techniques
* **Zero-shot** - directly instructs the model to perform a task without any additional examples or demonstrations to steer it.
* **Few-shot** - enable in-context learning where we provide demonstrations in the prompt to steer the model to better performance. The demonstrations serve as conditioning for subsequent examples where we would like the model to generate a response.
* **Chain-of-Thought (CoT)** - enables complex reasoning capabilities through intermediate reasoning steps. You can combine it with few-shot prompting to get better results on more complex tasks that require reasoning before responding.
* **Meta Prompting** - prioritizes (1) structure (format and pattern of problems and solutions over specific content), (2) syntax as a guiding template for the expected response, (3) abstracted examples as frameworks to illustrate the structure of problems and solutions without focusing on specific details;
  * draws from type theory to emphasize the categorization and logical arrangement of components in a prompt.
  * (Zhang et al., 2024)[https://arxiv.org/abs/2311.11482] reported that the advantages of Meta Prompting include token efficiency, fair comparison, and that the influence of specific examples is minimized (as it can be viewed as a form of zero-shot prompting).
* **Self-consistency** - aims to "replace the naive greedy decoding used in chain-of-thought prompting"; samples multiple, diverse reasoning paths through few-shot CoT and use the generations to select the most consistent answer.
* **Generated Knowledge** - includes the following steps:
  * Generate or establish a few knowledges.
  * Integrate the knowledge
  * Get a prediction

## Prompting Myths
1. Longer, Detailed Prompts equal better results. --> Structure matters more than length. A well-organized 50-word prompt often outperforms a rambling 500-word prompt while costing dramatically less to execute.
2. More Examples Always Help (Few-Shot Prompting) --> Advanced models like OpenAI’s o1 actually perform worse when given examples. They’re sophisticated enough to understand direct instructions and examples can introduce unwanted bias or noise.
3.  Perfect Wording Matters Most. Format beats content. XML tags, clear delimiters, and structured formatting provide more consistent improvements than perfect word choice. For Claude models specifically, XML formatting consistently provides a 15% performance boost compared to natural language formatting, regardless of the specific content.
4.  Chain-of-Thought Works for Everything.  Chain-of-thought is task-specific. It excels at math and logic but specialized approaches like Chain-of-Table work better for data analysis tasks.