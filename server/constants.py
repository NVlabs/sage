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
import os
import sys
SERVER_ROOT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(SERVER_ROOT_DIR, "results")

ROBOMIMIC_ROOT_DIR = "/home/hongchix/codes/robomimic" if os.path.exists("/home/hongchix/codes/robomimic") else "/home/hongchix/main/robomimic"
M2T2_ROOT_DIR = "/home/hongchix/codes/M2T2" if os.path.exists("/home/hongchix/codes/M2T2") else "/home/hongchix/main/M2T2"
PARTNET_ROOT_DIR = "/home/hongchix/Downloads/partnet-mobility"

MATFUSE_ROOT_DIR = "/home/hongchix/codes/matfuse-sd/src" if os.path.exists("/home/hongchix/codes/matfuse-sd/src") else "/home/hongchix/main/matfuse-sd/src"

PHYSICS_CRITIC_ENABLED = os.environ.get("PHYSICS_CRITIC_ENABLED", "true").lower() == "true"
SEMANTIC_CRITIC_ENABLED = os.environ.get("SEMANTIC_CRITIC_ENABLED", "true").lower() == "true"

print(f"PHYSICS_CRITIC_ENABLED: {PHYSICS_CRITIC_ENABLED}", file=sys.stderr)
print(f"SEMANTIC_CRITIC_ENABLED: {SEMANTIC_CRITIC_ENABLED}", file=sys.stderr)