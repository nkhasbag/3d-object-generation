#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
logger = logging.getLogger(__name__)

class GuardrailService:
    """Service for content filtering using Google ShieldGemma-2B."""
    
    def __init__(self, model_name="google/shieldgemma-2b"):
        """Initialize the guardrail service with ShieldGemma-2B."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.device = "cpu"
        
        # Safety policy definition
        self.safety_policy = """
    The prompt must not seek to generate harmful, abusive, hateful, sexually explicit, or otherwise inappropriate content targeting individuals or protected groups.
"""
        
    def load_model(self):
        """Load the ShieldGemma-2B model for content filtering."""
        try:
            if self.is_loaded:
                return True
                
            logger.info(f"Loading ShieldGemma-2B model: {self.model_name}")
            
            # Load the tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ).to(self.device)
            
            self.is_loaded = True
            
            logger.info("ShieldGemma-2B model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ShieldGemma-2B model: {e}")
            return False
    
    def is_dangerous(self, prompt, threshold):
        """Check if a prompt violates safety policies using ShieldGemma-2B."""
        try:
            if not self.load_model():
                logger.warning("Guardrail model not loaded, skipping content check")
                return False
            
            # Format the prompt for the model
            formatted_prompt = f"""You are a policy expert trying to help determine whether a user
    prompt is in violation of the defined safety policies.

    <start_of_turn>
    Human Question: {prompt.strip()}
    <end_of_turn>

    Our safety principle is defined in the below:

    {self.safety_policy.strip()}

    Does the human question violate the above principle? Your answer must start
    with 'Yes' or 'No'. And then walk through step by step to be sure we answer
    correctly.
    """
            
            # Get the device the model is on
            model_device = next(self.model.parameters()).device
            logger.debug(f"Model device: {model_device}")
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(model_device)
            logger.debug(f"Inputs device: {inputs['input_ids'].device}")
            
            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Extract the logits for the Yes and No tokens
            vocab = self.tokenizer.get_vocab()
            selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]

            # Convert these logits to a probability with softmax
            probabilities = softmax(selected_logits, dim=0)

            # Return probability of 'Yes' (violation)
            score = probabilities[0].item()
            
            logger.info(f"Content safety score for prompt: {score:.4f}")
            
            return score > threshold
            
        except Exception as e:
            logger.error(f"Error in content safety check: {e}")
            # In case of error, err on the side of caution and flag the content
            return True
    
    def check_prompt_safety(self, prompt, threshold=0.2):
        """Check if a prompt is safe for image generation."""
        try:
            is_violation = self.is_dangerous(prompt, threshold)
            
            if is_violation:
                logger.warning(f"2D prompt flagged as inappropriate: {prompt[:100]}...")
                return False, "PROMPT_CONTENT_FILTERED"
            else:
                logger.info(f"2D prompt passed safety check: {prompt[:100]}...")
                return True, "Content is safe for image generation"
                
        except Exception as e:
            logger.error(f"Error checking prompt safety: {e}")
            # In case of error, err on the side of caution
            return False, "Error in content safety check, please try again" 
        
