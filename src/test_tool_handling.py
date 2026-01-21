from open_r1.utils.multi_turn_handler import Tool, Conversations, Message, Prompt
from open_r1.utils import setup_project_logging

logger = setup_project_logging(log_file=None)


tool = Tool("crop_image", "Zoom in on the image based on the bounding box coordinates.",
            message=Message(image_position="last",
                            #text_message="Here is image <target_image> zoomed in at <bbox_2d>.",
                            text_message="\nHere is the cropped image (Image Size: <width>x<height>):",
                            text_fillers=["width", "height"]),
            parameter_descriptions={"bbox_2d": "coordinates for bounding box of the area you want to zoom in. Values should be within [0.0,1.0]."}
            )

conversations = Conversations(1)
conversations.add_message(Prompt(content=[{'text':None, 'type': 'image'},{'text': 'What is this image?', 'type': 'text'}], role="user",
                                 image_path="/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/dummy/test_img.jpeg"),
                          idx = 0)

logger.info(f"before: {conversations.get_full_for_hf_prep()}")

conversations.add_message(Prompt(content=[{'text':'Let me use the tool <tool_call>{"name": "crop_image", "arguments": {"bbox_2d": [0.2, 0.2, 0.8, 0.8], "target_image": 1}}</tool_call>', "type": "text"}], role="assistant"),
                          idx = 0)
logger.info(f"after: {conversations.get_full_for_hf_prep()}")
conversations.handle_tool_call("/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/dummy",0, tool)
logger.info(f"after tool: {conversations.get_full_for_hf_prep()}")