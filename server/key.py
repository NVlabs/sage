# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import sys
import subprocess

import tempfile
import os
import time
import hashlib

with open("key.json", "r") as f:
    key_dict = json.load(f)

ANTHROPIC_API_KEY = key_dict["ANTHROPIC_API_KEY"]

API_TOKEN = key_dict["API_TOKEN"]
API_URL_QWEN = key_dict["API_URL_QWEN"]
API_URL_OPENAI = key_dict["API_URL_OPENAI"]

API_URL_DICT = {
    "qwen": API_URL_QWEN,
    "openai": API_URL_OPENAI,
}

MODEL_DICT = key_dict["MODEL_DICT"]

SERVER_URL = key_dict["TRELLIS_SERVER_URL"]
FLUX_SERVER_URL = key_dict["FLUX_SERVER_URL"]

print("TRELLIS SERVER_URL: ", SERVER_URL, file=sys.stderr)



def slurm_job_id_to_port(job_id, port_start=8080, port_end=40000):
    """
    Hash-based mapping function to convert SLURM job ID to a port number.
    
    Args:
        job_id (str or int): SLURM job ID
        port_start (int): Starting port number (default: 8080)
        port_end (int): Ending port number (default: 40000)
    
    Returns:
        int: Mapped port number within the specified range
    """
    # Convert job_id to string if it's an integer
    job_id_str = str(job_id)
    
    # Create a hash of the job ID
    hash_obj = hashlib.md5(job_id_str.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    
    # Map to port range
    port_range = port_end - port_start + 1
    mapped_port = port_start + (hash_int % port_range)
    
    return mapped_port


import requests
from loguru import logger
from openai import OpenAI
from requests.auth import HTTPBasicAuth

TOKEN_VALIDITY_SECONDS = 4 * 60 * 60 - 60  # 4 hours minus an error threshold of 1 min

last_client_refresh = {
    "oai": 0.0,
    "anthropic": 0.0,
}


def is_client_valid(api_service: str) -> bool:
    """
    Checks whether the API client for the specified service (OpenAI or Anthropic)
    still has a valid access token.

    API tokens are valid for 4 hours (14400 seconds)
    """
    now = time.time()
    if api_service not in ["oai", "anthropic"]:
        return True

    if not last_client_refresh or now - last_client_refresh[api_service] > TOKEN_VALIDITY_SECONDS:
        logger.info("Client has an expired token, a new one needs to be generated")
        return False
    return True



def get_client_api_key(api_service: str) -> str:
    """
    Requests a new API access token for the specified corporate account (OpenAI or Anthropic).

    This function uses client credentials (API_CLIENT_ID and API_CLIENT_SECRET)
    which must be retrieved from LastPass and set as as environment variables
    """

    client_id = os.getenv("API_CLIENT_ID", key_dict["API_CLIENT_ID"])
    client_secret = os.getenv("API_CLIENT_SECRET", key_dict["API_CLIENT_SECRET"])

    url = key_dict["API_URL"]
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    scope_map = {"oai": "azureopenai-readwrite", "anthropic": "awsanthropic-readwrite"}
    data = {"grant_type": "client_credentials", "scope": scope_map[api_service]}

    response = requests.post(url, headers=headers, data=data, auth=HTTPBasicAuth(client_id, client_secret))
    if response.status_code != 200:
        raise RuntimeError("Error: Could not generate a Bearer API token, please try again")

    return response.json()["access_token"]


def setup_oai_client() -> OpenAI:
    """Set up corporate OpenAI client with an API key that needs to be refreshed every 4 hours"""
    last_client_refresh["oai"] = time.time()
    oai_api_key = get_client_api_key("oai")

    # Initialize OpenAI client with custom base URL and headers
    oai_client = OpenAI(
        api_key=oai_api_key,
        base_url="https://prod.api.nvidia.com/llm/v1/azure",
        default_headers={"dataClassification": "sensitive", "dataSource": "internet"},
    )
    return oai_client


# SERVER_URL = None
if __name__ == "__main__":
    print(SERVER_URL)