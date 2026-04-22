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
"""
Flux Image Generation Client
Similar to TrellisClient but for generating images using Flux model
"""

import requests
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from key import FLUX_SERVER_URL

try:
    from constants import SERVER_ROOT_DIR
except ImportError:
    # Fallback if imports fail
    SERVER_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FluxClient:
    def __init__(self, server_url=None):
        """
        Initialize Flux client
        Args:
            server_url: URL of the Flux server (default: from key.py or localhost:8080)
        """
        if server_url is None:
            server_url = FLUX_SERVER_URL

        self.server_url = server_url
    
    def health_check(self):
        """Check if the server is running"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            return response.json()
        except Exception as e:
            print(f"Health check failed: {e}", file=sys.stderr)
            return None
    
    def generate_image(self, prompt, height=1024, width=1024, guidance_scale=4.5, seed=None, output_file="generated_image.png"):
        """
        Generate image and save PNG file using two-phase protocol with retry logic
        
        Args:
            prompt: Text description of the image to generate
            height: Image height in pixels (default: 1024)
            width: Image width in pixels (default: 1024)
            guidance_scale: Guidance scale for generation (default: 4.5)
            seed: Random seed for reproducibility (optional)
            output_file: Path to save the generated image
            
        Returns:
            True if successful, False otherwise
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"\n⟳ Retry attempt {attempt}/{max_retries - 1}...", file=sys.stderr)
            
            try:
                print(f"Generating image for: '{prompt}'", file=sys.stderr)
                
                # Build request payload
                payload = {
                    "prompt": prompt,
                    "height": height,
                    "width": width,
                    "guidance_scale": guidance_scale
                }
                
                if seed is not None:
                    payload["seed"] = seed

                print("Waiting for server to be ready...", file=sys.stderr)

                # Wait for server to be ready
                while True:
                    health = self.health_check()
                    if health:
                        break
                    time.sleep(1)
                
                print("Server is ready. Submitting generation request...", file=sys.stderr)

                # Phase 1: Submit request and get acknowledgment
                total_trials = 0
                job_id = None
                
                while True:
                    try:
                        response = requests.post(
                            f"{self.server_url}/generate",
                            json=payload,
                            timeout=10  # Shorter timeout for acknowledgment
                        )
                        
                        if response.status_code == 202:
                            # Request acknowledged
                            ack_data = response.json()
                            job_id = ack_data.get('job_id')
                            print(f"✓ Request acknowledged. Job ID: {job_id}", file=sys.stderr)
                            print(f"Message: {ack_data.get('message', 'Processing started')}", file=sys.stderr)
                            break
                        else:
                            print(f"Server returned unexpected status: {response.status_code}", file=sys.stderr)
                            print(response.text, file=sys.stderr)

                    except Exception as e:
                        print(f"Error submitting request: {e}", file=sys.stderr)

                    total_trials += 1
                    if total_trials > 10:
                        raise Exception("Failed to submit request after 10 trials.")
                    
                    time.sleep(2)
                
                if not job_id:
                    raise Exception("Failed to get job ID from server.")
                
                # Phase 2: Poll for completion
                print("Waiting for generation to complete (this may take several minutes)...", file=sys.stderr)
                poll_count = 0
                max_polls = 200  # 200 seconds max
                
                while poll_count < max_polls:
                    try:
                        status_response = requests.get(
                            f"{self.server_url}/job/{job_id}",
                            timeout=10
                        )
                        
                        if status_response.status_code == 200:
                            # Job completed successfully
                            print("✓ Generation completed successfully!", file=sys.stderr)
                            with open(output_file, 'wb') as f:
                                f.write(status_response.content)
                            print(f"Image saved to: {output_file}", file=sys.stderr)
                            return True
                        
                        elif status_response.status_code == 202:
                            # Still processing
                            status_data = status_response.json()
                            if poll_count % 10 == 0:  # Print status every 10 polls
                                print(f"  Status: {status_data.get('status', 'processing')}... (poll {poll_count})", file=sys.stderr)
                        
                        elif status_response.status_code == 500:
                            # Job failed
                            error_data = status_response.json()
                            print(f"✗ Generation failed: {error_data.get('error', 'Unknown error')}", file=sys.stderr)
                            # Don't return False here, let it retry
                            break
                        
                        elif status_response.status_code == 404:
                            print(f"✗ Job not found on server", file=sys.stderr)
                            # Don't return False here, let it retry
                            break
                        
                    except Exception as e:
                        if poll_count % 10 == 0:
                            print(f"  Polling error (will retry): {e}", file=sys.stderr)
                    
                    poll_count += 1
                    time.sleep(1)  # Poll every 1 second
                
                # Timeout
                if poll_count >= max_polls:
                    print(f"✗ Timeout: Generation did not complete within {max_polls} seconds", file=sys.stderr)
                    # Don't return False here, let it retry
                    
            except Exception as e:
                print(f"Generation failed: {e}", file=sys.stderr)
                # Continue to next retry attempt
        
        # All retries exhausted
        print(f"✗ Failed after {max_retries} attempts", file=sys.stderr)
        return False
    
    def get_models(self):
        """Get list of available models"""
        try:
            response = requests.get(f"{self.server_url}/api/v1/models", timeout=10)
            return response.json()
        except Exception as e:
            print(f"Failed to get models: {e}", file=sys.stderr)
            return None


def generate_image_from_prompt(prompt, height=512, width=512, guidance_scale=4.5, seed=None, server_url=None):
    """
    Generate an image from a text prompt and return it as a PIL Image
    
    Args:
        prompt: Text description of the image to generate
        height: Image height in pixels (default: 1024)
        width: Image width in pixels (default: 1024)
        guidance_scale: Guidance scale for generation (default: 4.5)
        seed: Random seed for reproducibility (optional)
        server_url: URL of the Flux server (optional, uses default if not provided)
    
    Returns:
        PIL Image object if successful, None otherwise
    
    Example:
        >>> img = generate_image_from_prompt("A cozy living room with warm lighting")
        >>> if img:
        >>>     img.save("living_room.png")
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"\n⟳ Retry attempt {attempt}/{max_retries - 1} for generate_image_from_prompt...", file=sys.stderr)
        
        # Create client
        client = FluxClient(server_url=server_url)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name

        if attempt == 0:
            print(f"Generating image for: '{prompt}'", file=sys.stderr)
        
        try:
            # Generate image
            success = client.generate_image(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                seed=seed,
                output_file=temp_path
            )
            
            if success:
                # Load image from file
                img = Image.open(temp_path)
                # Create a copy in memory so we can delete the temp file
                img_copy = img.copy()
                img.close()
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                return img_copy
            else:
                # Clean up temporary file on failure
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                # Continue to next retry attempt
                if attempt < max_retries - 1:
                    print(f"Generation failed, will retry...", file=sys.stderr)
                    time.sleep(2)  # Wait before retrying
                
        except Exception as e:
            print(f"Error generating image: {e}", file=sys.stderr)
            # Clean up temporary file on exception
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            # Continue to next retry attempt
            if attempt < max_retries - 1:
                print(f"Will retry due to exception...", file=sys.stderr)
                time.sleep(2)  # Wait before retrying
    
    # All retries exhausted
    print(f"✗ generate_image_from_prompt failed after {max_retries} attempts", file=sys.stderr)
    return None


def test_flux_generation():
    """Test function to generate a sample image"""
    # Initialize client
    client = FluxClient()
    
    # Check server health
    health = client.health_check()
    if health:
        print("✓ Server is running and healthy", file=sys.stderr)
        print(f"GPU available: {health.get('gpu_available', 'Unknown')}", file=sys.stderr)
    else:
        print("✗ Cannot connect to server. Make sure it's running.", file=sys.stderr)
        return
    
    # Test image generation
    prompt = "A red sports car on a city street at sunset"
    output_path = os.path.join(SERVER_ROOT_DIR, "vis/flux_test_image.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    success = client.generate_image(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.5,
        seed=42,
        output_file=output_path
    )
    
    if success:
        print(f"✓ Image generation completed successfully!", file=sys.stderr)
        print(f"Image saved to: {output_path}", file=sys.stderr)
    else:
        print("✗ Image generation failed", file=sys.stderr)


def test_generate_image_from_prompt():
    """Test the generate_image_from_prompt function that returns PIL Image"""
    print("\n" + "="*60)
    print("Testing generate_image_from_prompt function")
    print("="*60)
    
    # Test 1: Basic generation
    print("\nTest 1: Generate and return PIL Image")
    prompt = "A modern minimalist living room with white walls and wooden floor"
    
    img = generate_image_from_prompt(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.5,
        seed=123
    )
    
    if img:
        print(f"✓ Image generated successfully!")
        print(f"  Image size: {img.size}")
        print(f"  Image mode: {img.mode}")
        
        # Save the image
        output_path = os.path.join(SERVER_ROOT_DIR, "vis/flux_pil_test.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        print(f"  Saved to: {output_path}")
        
        # Test 2: Generate multiple images
        print("\nTest 2: Generate multiple images with different prompts")
        prompts = [
            "A cozy bedroom with soft lighting",
            "A modern kitchen with marble countertops",
            "A luxurious bathroom with glass shower"
        ]
        
        images = []
        for i, p in enumerate(prompts):
            print(f"  Generating image {i+1}/{len(prompts)}: {p[:50]}...")
            img = generate_image_from_prompt(p, height=512, width=512, seed=100+i)
            if img:
                images.append(img)
                # Save each image
                save_path = os.path.join(SERVER_ROOT_DIR, f"vis/flux_pil_test_{i+1}.png")
                img.save(save_path)
                print(f"    ✓ Saved to: {save_path}")
            else:
                print(f"    ✗ Failed")
        
        print(f"\n✓ Generated {len(images)}/{len(prompts)} images successfully")
        
    else:
        print("✗ Image generation failed")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "pil":
        # Test the PIL Image function
        test_generate_image_from_prompt()
    else:
        # Default test
        test_flux_generation()