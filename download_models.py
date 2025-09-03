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

import torch
import gc

from diffusers import SanaSprintPipeline
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_models():
    try:
        logger.info("Starting model downloads...")
        
        # Download Sana Sprint model
        logger.info("Downloading Sana Sprint model...")
        sana_model = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16
        )
        del sana_model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Sana Sprint model downloaded successfully!")
        
        # Download Guardrail model (ShieldGemma-2B Safety Model)
        logger.info("Downloading ShieldGemma-2B safety model...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Download the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("google/shieldgemma-2b")
        model = AutoModelForCausalLM.from_pretrained(
            "google/shieldgemma-2b",
            torch_dtype=torch.bfloat16
        )
        
        del tokenizer, model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("ShieldGemma-2B safety model downloaded successfully!")
        
        logger.info("All models downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        return False

if __name__ == "__main__":
    download_models() 