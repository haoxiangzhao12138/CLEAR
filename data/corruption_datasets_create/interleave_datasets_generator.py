import re
import torch
import json
from copy import deepcopy
from PIL import Image
from data.data_utils import pil_img2rgb
from inferencer import InterleaveInferencer
# ================= 定义 Prompt =================
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
# 这是模拟工具执行后，系统回传给模型的提示
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
        temperature=0.7, # 稍微增加一点随机性，让推理过程更多样
        do_sample=True
    ):
        """
        生成单条数据。
        返回格式: Dict，包含完整的对话历史和状态。
        """
        # 1. 初始化 Context
        gen_context = self.init_gen_context()
        conversation_log = [] # 用于最后保存数据的列表

        # ---------------- ROUND 1: 输入 LQ 图片和问题 ----------------
        
        # 注入 System Prompt
        gen_context = self.update_context_text(RESTORATION_SYSTEM_PROMPT, gen_context)
        
        # 注入 LQ 图片 (只做 ViT 理解，不需要 VAE 生成)
        lq_input = self.vae_transform.resize_transform(pil_img2rgb(lq_image))
        gen_context = self.update_context_image(lq_input, gen_context, vae=False, vit=True)
        
        # 注入 User Question
        user_text = f"\nUser: {question}\nAssistant:" # 简单的 Chat 格式封装，视模型训练格式而定
        gen_context = self.update_context_text(user_text, gen_context)

        # 生成第一轮回复
        response_1 = self.gen_text(
            gen_context,
            max_length=max_tokens,
            do_sample=do_sample,
            temperature=temperature
        )
        
        # ---------------- ROUND 2: 分析与分支 ----------------

        tool_pattern = r"<tool_call>.*?</tool_call>"
        answer_pattern = r"<answer>.*?</answer>"

        has_tool_call = re.search(tool_pattern, response_1, re.DOTALL)
        has_final_answer = re.search(answer_pattern, response_1, re.DOTALL)

        # 分支 A: 模型认为不需要修复，直接回答了
        if has_final_answer and not has_tool_call:
            return {
                "status": "direct_answer",
                "question": question,
                "history": [
                    {"role": "user", "content": [{"type": "image", "source": "lq"}, {"type": "text", "text": question}]},
                    {"role": "assistant", "content": response_1}
                ]
            }

        # 分支 B: 模型调用了工具 (我们需要拼接 HQ 图片)
        elif has_tool_call:
            # 1. 将第一轮的 response_1 (包含 thinking 和 tool_call) 正式更新进 context
            # 注意：gen_text 只是生成了 tokens，我们需要手动把这些 tokens 变成 context 的一部分
            gen_context = self.update_context_text(response_1, gen_context)

            # 2. 注入 Observation 文本
            gen_context = self.update_context_text(OBSERVATION_PROMPT, gen_context)

            # 3. 注入 HQ 图片 (关键步骤：将清晰图作为“工具返回结果”拼接在后面)
            hq_input = self.vae_transform.resize_transform(pil_img2rgb(hq_image))
            gen_context = self.update_context_image(hq_input, gen_context, vae=False, vit=True)

            # 4. 生成第二轮回复 (模型看到清晰图后的反应 + 最终答案)
            response_2 = self.gen_text(
                gen_context,
                max_length=max_tokens,
                do_sample=do_sample,
                temperature=temperature
            )

            # 构造完整的数据记录
            return {
                "status": "restoration_used",
                "question": question,
                "history": [
                    # 第一轮
                    {"role": "user", "content": [{"type": "image", "source": "lq"}, {"type": "text", "text": question}]},
                    {"role": "assistant", "content": response_1},
                    # 工具返回 (Observation + HQ Image)
                    {"role": "tool", "content": [{"type": "text", "text": OBSERVATION_PROMPT}, {"type": "image", "source": "hq"}]},
                    # 第二轮 (最终回答)
                    {"role": "assistant", "content": response_2}
                ]
            }

        else:
            # 异常情况 (比如生成了一半断了)
            return {"status": "failed", "raw_output": response_1}



