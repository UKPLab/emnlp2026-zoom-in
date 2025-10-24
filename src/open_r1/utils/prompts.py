
def get_question_template(task_type: str):
    match task_type:
        case "rec":
            return "{Question} \n First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
        case "no_think":
            return "{Question} \n First output the thinking process and then output the final answer in <answer> </answer> tags."
        case "no_think_tool":
            return "{Question} \n First output the thinking process. During your thinking, you can use one of the tools if you think it might be helpful to answer the question. Once you are done, output the final answer in <answer> </answer> tags."
        case "no_think_tool_enforce":
            return "{Question} \n First output the thinking process. During your reasoning, you must use the tool to answer the question. Once you are done, output the final answer in <answer> </answer> tags."
        case "pr_adapted":
            return "{Question}\n\nGuidelines: Understand the given visual information and the user query. Determine if it is beneficial to employ the given visual operations (tools). For an image, we can look closer by `{tool_name}`. Reason with the visual information step by step, and put your final answer within \\boxed{}."
        case "pr_adapted_exploration":
            return "Explore the visual information present in the image. Determine if you can see more by employing the given visual operations (tools). For an image, we can look closer by `{tool_name}`. Describe the input image and the output images of the tool use (where applicable) in detail."
        case "pr_original":
            return "{Question}\n\nGuidelines: Understand the given visual information and the user query. Determine if it is beneficial to employ the given visual operations (tools). For a video, we can look closer by `{tool_name2}`. For an image, we can look closer by `{tool_name1}`. Reason with the visual information step by step, and put your final answer within \\boxed{}."
        case "no_tool":
            return "{Question} \n Please think step by step, and put your final answer within \\boxed{}."
        case _:
            return "{Question} \n First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
