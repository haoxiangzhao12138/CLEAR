import re
import torch
import json
from copy import deepcopy
from PIL import Image
from data.data_utils import pil_img2rgb
from inferencer import InterleaveInferencer
# ================= Define Prompts =================
RESTORATION_SYSTEM_PROMPT = """You are a specialized multimodal agent adept at handling low-quality or corrupted visual inputs. Your goal is to answer questions based on an input image that may suffer from degradations (e.g., blur, noise, occlusion, low resolution).

# Tools
You have access to a special token: **<image_restore>**.
**Description:** Image restoration tool. It takes the current corrupted image as input and performs enhancement operations (such as deblurring, denoising, super-resolution, or inpainting) to return a high-quality, clearer version of the image. Use this tool ONLY when the image corruption prevents you from seeing the specific details required to answer the user's question.

# Instruction
1.  **Always start with a <think> tag.** Inside this tag, you must conduct a step-by-step reasoning process:
    * **Analyze Image Quality:** Identify the type and severity of the corruption (e.g., "The image is heavily blurred," or "There is slight noise but the subject is visible").
    * **Assess Information Sufficiency:** Compare the image quality against the specific requirements of the user's question.
        * **Case A (Direct Answer):** If the image is corrupted but the answer is still visually evident (e.g., the user asks for the dominant color of a blurry car), **do not** use the tool. Proceed to answer.
        * **Case B (Need Restoration):** If the corruption makes it impossible to extract the necessary information with confidence (e.g., the user asks for text on a blurry sign), you **must** use the restoration token.
2.  **Tool Execution:**
    * If you choose to use the tool, simply output the <image_restore> token.
    * After receiving the tool result (the restored image), analyze the new visual information in the next turn's <think> block.
3.  **Final Answer:**
    * When no further tools are needed, provide your final response in the <answer> tag.
    * The answer should be natural and direct.
4.  **Stop Token:** Put the `eos` tag after the <image_restore> token to wait for the tool execution.

# Response Format Examples
**Scenario 1: Image is too blurry to answer -> Call Tool**
<think>The user is asking for the license plate number. I can see the car clearly, but the license plate area is heavily affected by motion blur, making the text indistinguishable. Since I cannot read the specific characters required to answer the question, I need to restore the image to recover these details.</think>
<image_restore>

**Scenario 2: Image is corrupted but answer is visible -> Direct Answer**
<think>The user asks if there is a dog in the picture. Although the image has some "salt and pepper" noise, the silhouette and main features of a Golden Retriever are clearly visible in the center. The noise does not hinder my ability to classify the object. I can answer directly without restoration.</think>
<answer> Yes, there is a dog in the picture. It appears to be a Golden Retriever sitting on the grass. </answer>
"""
# This simulates the system response sent back to the model after tool execution
OBSERVATION_PROMPT = "\nObservation: The image has been restored. Here is the high-quality version.\n"

class RestorationDataGenerator(InterleaveInferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def generate_sample(
        self,
        lq_image: Image.Image,
        hq_image: Image.Image,
        question: str,
        max_tokens=1024,
        temperature=0.7, # Slightly increase randomness for more diverse reasoning
        do_sample=True
    ):
        """
        Generate a single data sample.
        Returns a Dict containing the complete conversation history and status.
        """
        # 1. Initialize Context
        gen_context = self.init_gen_context()
        conversation_log = [] # List for saving the final data

        # ---------------- ROUND 1: Input LQ image and question ----------------

        # Inject System Prompt
        gen_context = self.update_context_text(RESTORATION_SYSTEM_PROMPT, gen_context)
        
        # Inject LQ image (only for ViT understanding, no VAE generation needed)
        lq_input = self.vae_transform.resize_transform(pil_img2rgb(lq_image))
        gen_context = self.update_context_image(lq_input, gen_context, vae=False, vit=True)
        
        # Inject User Question
        user_text = f"\nUser: {question}\nAssistant:" # Simple chat format wrapper, may vary depending on model training format
        gen_context = self.update_context_text(user_text, gen_context)

        # Generate first round response
        response_1 = self.gen_text(
            gen_context,
            max_length=max_tokens,
            do_sample=do_sample,
            temperature=temperature
        )
        
        # ---------------- ROUND 2: Analysis and branching ----------------

        tool_pattern = r"<tool_call>.*?</tool_call>"
        answer_pattern = r"<answer>.*?</answer>"

        has_tool_call = re.search(tool_pattern, response_1, re.DOTALL)
        has_final_answer = re.search(answer_pattern, response_1, re.DOTALL)

        # Branch A: Model decided restoration is not needed, answered directly
        if has_final_answer and not has_tool_call:
            return {
                "status": "direct_answer",
                "question": question,
                "history": [
                    {"role": "user", "content": [{"type": "image", "source": "lq"}, {"type": "text", "text": question}]},
                    {"role": "assistant", "content": response_1}
                ]
            }

        # Branch B: Model called the tool (we need to append the HQ image)
        elif has_tool_call:
            # 1. Formally update response_1 (containing thinking and tool_call) into the context
            # Note: gen_text only generates tokens; we need to manually make them part of the context
            gen_context = self.update_context_text(response_1, gen_context)

            # 2. Inject Observation text
            gen_context = self.update_context_text(OBSERVATION_PROMPT, gen_context)

            # 3. Inject HQ image (key step: append the clean image as the “tool return result”)
            hq_input = self.vae_transform.resize_transform(pil_img2rgb(hq_image))
            gen_context = self.update_context_image(hq_input, gen_context, vae=False, vit=True)

            # 4. Generate second round response (model's reaction after seeing the clean image + final answer)
            response_2 = self.gen_text(
                gen_context,
                max_length=max_tokens,
                do_sample=do_sample,
                temperature=temperature
            )

            # Build the complete data record
            return {
                "status": "restoration_used",
                "question": question,
                "history": [
                    # Round 1
                    {"role": "user", "content": [{"type": "image", "source": "lq"}, {"type": "text", "text": question}]},
                    {"role": "assistant", "content": response_1},
                    # Tool return (Observation + HQ Image)
                    {"role": "tool", "content": [{"type": "text", "text": OBSERVATION_PROMPT}, {"type": "image", "source": "hq"}]},
                    # Round 2 (final answer)
                    {"role": "assistant", "content": response_2}
                ]
            }

        else:
            # Abnormal case (e.g., generation was cut off midway)
            return {"status": "failed", "raw_output": response_1}



