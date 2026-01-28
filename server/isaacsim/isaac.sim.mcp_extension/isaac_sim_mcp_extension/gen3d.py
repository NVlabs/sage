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
import asyncio
import os
import time
import json
import requests
import zipfile
import shutil
from pathlib import Path

class Beaver3d:
    def __init__(self):
        """Initialize Beaver3d with model name and API key from environment variables"""
        self.api_key = os.environ.get("ARK_API_KEY")
        self.model_name = os.environ.get("BEAVER3D_MODEL")
        self._working_dir = Path(os.environ.get("USD_WORKING_DIR", "/tmp/usd"))
        self.base_url = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks")
        
        if not self.api_key:
            raise Exception("ARK_API_KEY environment variable not set, Beaver3D service is not available untill ARK_API_KEY is set")
        if not self.model_name:
            raise Exception("BEAVER3D_MODEL environment variable not set, Beaver3D service is not available untill BEAVER3D_MODEL is set")
        
    
    def _get_headers(self):
        """Get request headers with authorization"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _download_files_for_completed_task(self, task_id, file_url):
        """
        Process a completed task by downloading and extracting the result file
        
        Args:
            task_id (str): The task ID
            file_url (str): URL to download the result file
            
        Returns:
            str: Path to the extracted task directory
        """
        # Create directories if they don't exist
        tmp_dir = self._working_dir
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        task_dir = tmp_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        zip_path = tmp_dir / f"{task_id}.zip"
        response = requests.get(file_url)
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(task_dir)
        
        return str(task_dir)
    
    def monitor_task_status(self, task_id):
        """Monitor task status until succeeded, then download and extract the result file"""
        task_url = f"{self.base_url}/{task_id}"
        elapsed_time_in_seconds = 0
        estimated_time_in_seconds = 75 # 75 seconds is the estimated time for a high subdivision level USD model
        while True:
            response = requests.get(task_url, headers=self._get_headers())
            if response.status_code != 200:
                raise Exception(f"Error fetching task status: {response.text}")
                
            task_data = response.json()
            status = task_data.get("status")
            
            if status == "succeeded":
                file_url = task_data.get("content", {}).get("file_url")
                if not file_url:
                    raise Exception("No file URL found in the response")
                
                return self._download_files_for_completed_task(task_id, file_url)
            
            if status == "failed":
                raise Exception(f"Task failed: {task_data}")
            elif status == "running":
                # Assuming generation takes about 70s total, calculate an estimated completion percentage
                # Each iteration is 5s, so after 70s we should be at 100%
                
                completion_ratio = min(100, int((elapsed_time_in_seconds / estimated_time_in_seconds) * 100))
                print(f"Task {task_id} is generating. Progress: {completion_ratio}% complete. Waiting for completion...")
            
            # Sleep for 5 seconds before checking again
            time.sleep(5)
            elapsed_time_in_seconds += 5
    
    async def monitor_task_status_async(self, task_id, on_complete_callback=None):
        """
        Asynchronously monitor task status until succeeded, then download and extract the result file
        
        Args:
            task_id (str): The task ID to monitor
            on_complete (callable, optional): Callback function to call when task completes
                                             with the task directory as argument
        
        Returns:
            str: Path to the extracted task directory
        """
        import asyncio
        
        task_url = f"{self.base_url}/{task_id}"
        elapsed_time_in_seconds = 0
        estimated_time_in_seconds = 75  # 75 seconds is the estimated time for a high subdivision level USD model
        
        while True:
            response = requests.get(task_url, headers=self._get_headers())
            if response.status_code != 200:
                raise Exception(f"Error fetching task status: {response.text}")
                
            task_data = response.json()
            status = task_data.get("status")
            
            if status == "succeeded":
                file_url = task_data.get("content", {}).get("file_url")
                if not file_url:
                    raise Exception("No file URL found in the response")
                
                result_path = self._download_files_for_completed_task(task_id, file_url)
                
                # Call the callback if provided
                if on_complete_callback and callable(on_complete_callback):
                    on_complete_callback(task_id, status, result_path)
                
                return result_path
            
            if status == "failed":
                raise Exception(f"Task failed: {task_data}")
            elif status == "running":
                # Calculate an estimated completion percentage
                completion_ratio = min(100, int((elapsed_time_in_seconds / estimated_time_in_seconds) * 100))
                print(f"Task {task_id} is generating. Progress: {completion_ratio}% complete. Waiting for completion...")
            
            # Asynchronously sleep for 5 seconds before checking again
            await asyncio.sleep(5)
            elapsed_time_in_seconds += 5
    
    def generate_3d_from_text(self, text_prompt):
        """Generate a 3D model from text input and return the task ID"""
        # Add default options for USD generation if not already present
        if "--subdivision_level" not in text_prompt:
            text_prompt += " --subdivision_level high"
        if "--fileformat" not in text_prompt:
            text_prompt += " --fileformat usd"
            
        payload = {
            "model": self.model_name,
            "content": [
                {
                    "type": "text",
                    "text": text_prompt
                }
            ]
        }
        
        response = requests.post(
            self.base_url,
            headers=self._get_headers(),
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Error generating 3D model: {response.text}")
        
        return response.json().get("id")
    
    def generate_3d_from_image(self, image_url, text_options="--subdivision_level high --fileformat usd --watermark true"):
        """Generate a 3D model from an image URL and return the task ID"""
        """
        Generate a 3D model from an image URL and return the task ID
        
        Args:
            image_url (str): URL of the image to generate 3D model from
            text_options (str): Additional options for the generation
            
        Returns:
            str: Task ID for the generation job
        """
        # Add default options for USD generation if not already present
        if "--subdivision_level" not in text_options:
            text_options += " --subdivision_level high"
        if "--fileformat" not in text_options:
            text_options += " --fileformat usd"
        if "--watermark" not in text_options:
            text_options += " --watermark true"
        payload = {
            "model": self.model_name,
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": text_options
                }
            ]
        }
        
        response = requests.post(
            self.base_url,
            headers=self._get_headers(),
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Error generating 3D model from image: {response.text}")
        
        return response.json().get("id")


def main():
    """Main function to test the Beaver3d class"""
    try:
        # Initialize the Beaver3d class
        beaver = Beaver3d()
        
        # Generate a 3D model from text
        text_prompt = "A gothic castle with 4 towers surrounding a central tower, inspired by Notre-Dame de Paris --subdivision_level high --fileformat usd"
        task_id = beaver.generate_3d_from_text(text_prompt)
        print(f"3D model generation task started with ID: {task_id}")
        
        # Monitor the task and download the result
        result_path = beaver.monitor_task_status(task_id)
        print(f"3D model downloaded to: {result_path}")
        
        # Generate a 3D model from an image
        image_url = "https://lvlecheng.tos-cn-beijing.volces.com/chore/apple.jpg"
        task_id = beaver.generate_3d_from_image(image_url)
        print(f"3D model generation from image task started with ID: {task_id}")
        
        # Monitor the task and download the result
        result_path = beaver.monitor_task_status(task_id)
        print(f"3D model from image downloaded to: {result_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

async def test_async():
    # Test initialization
        beaver = Beaver3d()
        assert beaver.api_key, "API key should be set"
        assert beaver.model_name, "Model name should be set"

        # Test async task monitoring with callback
        def call_back_fn(task_id, status=None, result_path=None):
            print(f"Callback invoked: Task {task_id} status is {status}")
            if result_path:
                print(f"Callback received result path: {result_path}")
            return True
        
        # Test async monitoring
        async_task_id = beaver.generate_3d_from_text("A simple chair")
        assert async_task_id, "Async task ID should be returned"
        print(f"Starting async monitoring for task ID: {async_task_id}")
        
        # Monitor the task asynchronously with callback
        result_path = await beaver.monitor_task_status_async(async_task_id, on_complete_callback=call_back_fn)
        print(f"Async monitoring initiated for task ID: {async_task_id}")
        print(f"Async monitoring completed, result path: {result_path}")
        print("Async test completed")
        return result_path

def test():
    """Unit test for the Beaver3d class"""
    try:
        # Test initialization
        beaver = Beaver3d()
        assert beaver.api_key, "API key should be set"
        assert beaver.model_name, "Model name should be set"

        
        
        # Test text generation
        text_prompt = "An fresh apple in red --subdivision_level high --fileformat usd"
        task_id = beaver.generate_3d_from_text(text_prompt)
        assert task_id, "Task ID should be returned"
        print(f"Text generation test passed, task ID: {task_id}")
        result_path = beaver.monitor_task_status(task_id)
        result_path_obj = Path(result_path)
        assert result_path_obj.exists(), f"Downloaded file does not exist at {result_path}"
        assert result_path_obj.is_dir(), f"Expected directory at {result_path}"
        
        # Check if there are USD files in the extracted directory
        usd_files = list(result_path_obj.glob("*.usd")) + list(result_path_obj.glob("*.usda")) + list(result_path_obj.glob("*.usdc"))
        assert len(usd_files) > 0, f"No USD files found in {result_path}"
        print(f"Verified: USD file successfully downloaded and extracted to {result_path}")
        
        # Test text generation with Chinese text
        chinese_text_prompt = "一个哥特式的城堡,参考巴黎圣母院 "
        task_id = beaver.generate_3d_from_text(chinese_text_prompt)
        assert task_id, "Task ID should be returned"
        print(f"Chinese text generation test passed, task ID: {task_id}")
        result_path = beaver.monitor_task_status(task_id)
        result_path_obj = Path(result_path)
        assert result_path_obj.exists(), f"Downloaded file does not exist at {result_path}"
        assert result_path_obj.is_dir(), f"Expected directory at {result_path}"
        
        # Check if there are USD files in the extracted directory
        usd_files = list(result_path_obj.glob("*.usd")) + list(result_path_obj.glob("*.usda")) + list(result_path_obj.glob("*.usdc"))
        assert len(usd_files) > 0, f"No USD files found in {result_path}"
        print(f"Verified: USD file from Chinese text successfully downloaded and extracted to {result_path}")
        
        # Test image generation
        image_url = "https://lvlecheng.tos-cn-beijing.volces.com/chore/apple.jpg"
        task_id = beaver.generate_3d_from_image(image_url)
        assert task_id, "Task ID should be returned"
        print(f"Image generation test passed, task ID: {task_id}")
        result_path = beaver.monitor_task_status(task_id)
        result_path_obj = Path(result_path)
        assert result_path_obj.exists(), f"Downloaded file does not exist at {result_path}"
        assert result_path_obj.is_dir(), f"Expected directory at {result_path}"
        
        # Check if there are USD files in the extracted directory
        usd_files = list(result_path_obj.glob("*.usd")) + list(result_path_obj.glob("*.usda")) + list(result_path_obj.glob("*.usdc"))
        assert len(usd_files) > 0, f"No USD files found in {result_path}"
        print(f"Verified: USD file successfully downloaded and extracted to {result_path}")
        
        
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")


if __name__ == "__main__":
    

    asyncio.run(test_async())
    # task = asyncio.create_task(test_async())
    test()
    
    
    # Schedule the test to run in the background
    # task = asyncio.create_task(run_test_in_background())
    # task.add_done_callback(lambda _: print(f"Test completed: {result_path}"))

    # print("Test scheduled to run in background")
    # while result_path is None:
    #     sleep(1)
    # print(f"Async test completed, result path: {result_path}")
    
    
