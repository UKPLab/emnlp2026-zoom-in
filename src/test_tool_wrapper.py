from open_r1.utils.tools import Tool, Message, TOOL_CONFIGS
from open_r1.utils.prompts import get_question_template
from pprint import pprint
import json



def get_tool(tool_config, prompt_type, max_pixels, min_pixels, tool_bbox_type):
    tool_args = TOOL_CONFIGS[tool_config if tool_config is not None else "no_tool"]
    #print(f"tool_args: {tool_args}")
    question_prompt = get_question_template(task_type=prompt_type if
    prompt_type is not None else tool_args["prompt_type"]).replace("{tool_name}", tool_args["tool_name"])

    if tool_config != "no_tool":
        tools = Tool(name=tool_args["tool_name"],
                     template_name=tool_args["tool_template"],
                     json_customization=tool_args["tool_json_customization"],
                     message=Message(tool_args["tool_message_image_pos"],
                                     tool_args["tool_message_text_message"],
                                     tool_args["tool_message_text_fillers"]),

                     tool_hparams={"max_pixels": max_pixels,
                                   "min_pixels": min_pixels,
                                   "bbox_type": tool_bbox_type})
    else:
        tools = None

    return tools, question_prompt

if __name__ == "__main__":
    #tool_config = None
    #prompt_type = None
    for tool_config in ["PR_zoom_in", "PR_zoom_in_with_hint", "zoom_in_absolute", "zoom_in_relative", "no_tool", "select_frames"]:
        #tool_config = "PR_zoom_in_very_old"
        prompt_type = None

        max_pixels = None
        min_pixels = None
        tool_bbox_type = None
        tools, question_prompt = get_tool(tool_config,
                                        prompt_type,
                                        max_pixels,
                                        min_pixels,
                                        tool_bbox_type)
        print(f"tool config: {tool_config}")
        if tools is not None:
            print(json.dumps(tools.tool_dict, indent=4))
        else:
            print(f"tools is None")
        print("\n")
        #pprint(tools.tool_dict, width=1000000)