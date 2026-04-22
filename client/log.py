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
import os
from datetime import datetime

def _convert_organized_to_conversation(organized_conversation):
    """
    Convert organized_conversation format to conversation_history format.
    
    Args:
        organized_conversation: List of organized conversation items with type field
        
    Returns:
        list: Conversation history in expected format
    """
    conversation = []
    current_assistant_content = []
    
    for item in organized_conversation:
        item_type = item.get('type', 'unknown')
        
        if item_type == 'user_message':
            # Flush any pending assistant content
            if current_assistant_content:
                conversation.append({
                    'role': 'assistant',
                    'content': current_assistant_content
                })
                current_assistant_content = []
            
            # Add user message
            conversation.append({
                'role': 'user',
                'content': item.get('content', '')
            })
        
        elif item_type == 'reasoning':
            # Extract reasoning/thinking content
            content_text = item.get('content', '')
            raw_content = item.get('raw_content', [])
            
            # Try to parse the reasoning text
            thinking_text = content_text
            
            # If content is a string representation of Content object, parse it
            if 'Content(text=' in thinking_text:
                import re
                # Updated regex to handle escaped quotes: match either double-quoted or single-quoted strings
                # For single quotes, we need to match until we find an unescaped single quote
                # Pattern: ' followed by (anything except ', OR \' escaped quote)*, followed by unescaped '
                text_match = re.search(r"Content\(text=(?:\"((?:[^\"]|\\\")*)\")|(?:'((?:[^'\\]|\\.)*)')", thinking_text, re.DOTALL)
                if text_match:
                    thinking_text = text_match.group(1) or text_match.group(2)
                    # Unescape the text - handle all common escape sequences
                    thinking_text = thinking_text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
                    thinking_text = thinking_text.replace('\\\'', "'").replace('\\"', '"').replace('\\\\', '\\')
            
            # Add as thinking content
            current_assistant_content.append({
                'type': 'thinking',
                'thinking': thinking_text
            })
        
        elif item_type == 'function_call':
            # Add function call as tool use
            tool_name = item.get('name', 'unknown')
            arguments = item.get('arguments', '{}')
            call_id = item.get('call_id', 'unknown')
            
            # Parse arguments if string
            import json
            try:
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
            except:
                pass
            
            current_assistant_content.append({
                'type': 'tool_use',
                'id': call_id,
                'name': tool_name,
                'input': arguments
            })
        
        elif item_type == 'tool_result':
            # Add tool result from organized_conversation (new format)
            content = item.get('content', '')
            tool_call_id = item.get('tool_call_id', 'unknown')
            is_error = item.get('error', False)
            
            current_assistant_content.append({
                'type': 'tool_result',
                'tool_use_id': tool_call_id,
                'content': content,
                'is_error': is_error
            })
        
        elif item_type == 'function_output':
            # Add function output as tool result (legacy format)
            output = item.get('output', '')
            call_id = item.get('call_id', 'unknown')
            is_error = item.get('error', False)
            
            current_assistant_content.append({
                'type': 'tool_result',
                'tool_use_id': call_id,
                'content': output,
                'is_error': is_error
            })
        
        elif item_type == 'assistant_response' or item_type == 'ai_response':
            # Flush any pending assistant content first
            if current_assistant_content:
                conversation.append({
                    'role': 'assistant',
                    'content': current_assistant_content
                })
                current_assistant_content = []
            
            # Handle AI response with potential reasoning
            assistant_content = []
            
            # Add reasoning if available
            if 'reasoning' in item:
                reasoning_value = item.get('reasoning', [])
                # If it's a string, treat it as a single item (not iterate over characters)
                if isinstance(reasoning_value, str):
                    reasoning_items = [reasoning_value]
                else:
                    reasoning_items = reasoning_value
                
                for reasoning_item in reasoning_items:
                    reasoning_text = str(reasoning_item)
                    # Parse if it's a Content object representation
                    if 'Content(text=' in reasoning_text:
                        import re
                        # Updated regex to handle escaped quotes properly
                        text_match = re.search(r"Content\(text=(?:\"((?:[^\"]|\\\")*)\")|(?:'((?:[^'\\]|\\.)*)')", reasoning_text, re.DOTALL)
                        if text_match:
                            reasoning_text = text_match.group(1) or text_match.group(2)
                            # Unescape the text - handle all common escape sequences
                            reasoning_text = reasoning_text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
                            reasoning_text = reasoning_text.replace('\\\'', "'").replace('\\"', '"').replace('\\\\', '\\')
                    
                    assistant_content.append({
                        'type': 'thinking',
                        'thinking': reasoning_text
                    })
            
            # Add text content if available
            content = item.get('content', '')
            if content:
                assistant_content.append({
                    'type': 'text',
                    'text': content
                })
            
            # Add to conversation if we have content
            if assistant_content:
                conversation.append({
                    'role': 'assistant',
                    'content': assistant_content
                })
    
    # Flush any remaining assistant content
    if current_assistant_content:
        conversation.append({
            'role': 'assistant',
            'content': current_assistant_content
        })
    
    return conversation

def _convert_input_list_to_conversation(input_list):
    """
    Convert input_list format to conversation_history format.
    
    Args:
        input_list: List of various message types from input_list format
        
    Returns:
        list: Conversation history in expected format
    """
    conversation = []
    current_assistant_content = []
    
    for item in input_list:
        if isinstance(item, dict):
            # Handle user messages and function call outputs
            if item.get('role') == 'user':
                # Flush any pending assistant content
                if current_assistant_content:
                    conversation.append({
                        'role': 'assistant',
                        'content': current_assistant_content
                    })
                    current_assistant_content = []
                
                conversation.append(item)
            
            elif item.get('role') == 'assistant':
                # Handle our enhanced assistant messages
                # Flush any pending assistant content first
                if current_assistant_content:
                    conversation.append({
                        'role': 'assistant', 
                        'content': current_assistant_content
                    })
                    current_assistant_content = []
                
                # Start new assistant content
                assistant_content = []
                
                # Add reasoning if available
                if 'reasoning' in item:
                    for reasoning_item in item['reasoning']:
                        reasoning_text = str(reasoning_item)
                        assistant_content.append({
                            'type': 'thinking',
                            'thinking': reasoning_text
                        })
                
                # Add main content
                if 'content' in item and item['content']:
                    assistant_content.append({
                        'type': 'text',
                        'text': item['content']
                    })
                
                # Add to conversation
                if assistant_content:
                    conversation.append({
                        'role': 'assistant',
                        'content': assistant_content
                    })
            
            elif item.get('type') == 'function_call_output':
                # Add as tool result content
                current_assistant_content.append({
                    'type': 'tool_result',
                    'tool_use_id': item.get('call_id', ''),
                    'content': item.get('output', ''),
                    'is_error': item.get('error', False)
                })
        
        elif isinstance(item, str):
            # Parse string representations of function calls and reasoning
            if item.startswith('ResponseFunctionToolCall'):
                # Extract function call information
                try:
                    # Parse the function call string - this is a simplified parser
                    import re
                    name_match = re.search(r"name='([^']+)'", item)
                    # Handle multi-line JSON arguments
                    args_match = re.search(r"arguments='({.*?})'", item, re.DOTALL)
                    id_match = re.search(r"call_id='([^']+)'", item)
                    
                    if name_match and args_match and id_match:
                        tool_name = name_match.group(1)
                        arguments_str = args_match.group(1)
                        tool_id = id_match.group(1)
                        
                        # Clean up the arguments string
                        arguments_str = arguments_str.replace('\\n', '\n').replace('\\"', '"')
                        
                        # Parse arguments JSON
                        try:
                            arguments = json.loads(arguments_str)
                        except:
                            arguments = arguments_str
                        
                        current_assistant_content.append({
                            'type': 'tool_use',
                            'id': tool_id,
                            'name': tool_name,
                            'input': arguments
                        })
                except:
                    # Fallback: add as text content
                    current_assistant_content.append({
                        'type': 'text',
                        'text': item
                    })
            
            elif item.startswith('ResponseReasoningItem'):
                # Extract reasoning content
                try:
                    import re
                    # Look for the 'text' content within the reasoning item
                    text_match = re.search(r"'text': '(.*?)', 'type': 'reasoning_text'", item, re.DOTALL)
                    if text_match:
                        thinking_text = text_match.group(1)
                        # Clean up escape sequences
                        thinking_text = thinking_text.replace("\\'", "'").replace("\\n", "\n").replace("\\\\", "\\")
                        current_assistant_content.append({
                            'type': 'thinking',
                            'thinking': thinking_text
                        })
                    else:
                        # Try alternative format - just extract the reasoning text
                        reasoning_text = item.replace('ResponseReasoningItem: ', '')
                        current_assistant_content.append({
                            'type': 'thinking',
                            'thinking': reasoning_text
                        })
                except:
                    # Fallback: add as text content
                    current_assistant_content.append({
                        'type': 'text',
                        'text': item
                    })
            else:
                # Other string content
                current_assistant_content.append({
                    'type': 'text',
                    'text': item
                })
    
    # Flush any remaining assistant content
    if current_assistant_content:
        conversation.append({
            'role': 'assistant',
            'content': current_assistant_content
        })
    
    return conversation

def chat_log_to_html(log_file):
    """
    Convert client chat logs to HTML format with inline CSS styling.
    
    Args:
        log_file: Path to the JSON log file
        
    Returns:
        str: HTML content of the chat logs
    """
    try:
        # Read the JSON log file
        if not os.path.exists(log_file):
            return f"<html><body><h1>Log file not found: {log_file}</h1></body></html>"
            
        with open(log_file, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
            
        # Handle both conversation_history and input_list formats
        if not isinstance(log_data, dict):
            return "<html><body><h1>Invalid client log format</h1></body></html>"
        
        # Check for either conversation_history or input_list
        if 'conversation_history' not in log_data and 'input_list' not in log_data and 'raw_input_list' not in log_data:
            return "<html><body><h1>Invalid client log format - missing conversation data</h1></body></html>"
            
    except Exception as e:
        return f"<html><body><h1>Error reading log file: {str(e)}</h1></body></html>"
    
    # Extract session metadata
    session_start = log_data.get('session_start', 'Unknown')
    session_end = log_data.get('session_end', 'Unknown')
    connected_servers = log_data.get('connected_servers', {})
    
    # Handle conversation data - prioritize organized_conversation (best format)
    tool_uses_count = 0
    if 'organized_conversation' in log_data:
        # Use organized_conversation directly - it's already well-structured
        organized_conv = log_data.get('organized_conversation', [])
        conversation_history = _convert_organized_to_conversation(organized_conv)
        # Count tool uses from organized_conversation (tool_result entries indicate tool uses)
        tool_uses_count = sum(1 for item in organized_conv if item.get('type') == 'tool_result')
    elif 'conversation_history' in log_data:
        conversation_history = log_data.get('conversation_history', [])
    else:
        # Convert input_list format to conversation_history format
        input_list = log_data.get('input_list', log_data.get('raw_input_list', []))
        conversation_history = _convert_input_list_to_conversation(input_list)
    
    total_messages = log_data.get('total_messages', len(conversation_history))
    
    # Calculate session duration
    try:
        if session_start != 'Unknown' and session_end != 'Unknown':
            start_dt = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(session_end.replace('Z', '+00:00'))
            duration = end_dt - start_dt
            duration_str = str(duration).split('.')[0]  # Remove microseconds
        else:
            duration_str = 'Unknown'
    except:
        duration_str = 'Unknown'
    
    # HTML template with inline CSS
    log_filename = os.path.basename(log_file)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MCP Client Chat Log</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                font-size: 2.2em;
            }}
            .header p {{
                margin: 5px 0;
                opacity: 0.9;
            }}
            .session-info {{
                background: #f8f9fa;
                padding: 20px;
                border-bottom: 1px solid #eee;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 15px;
            }}
            .info-item {{
                background: white;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #4CAF50;
            }}
            .info-label {{
                font-weight: bold;
                color: #495057;
                margin-bottom: 5px;
            }}
            .info-value {{
                color: #6c757d;
                word-break: break-all;
            }}
            .servers-section {{
                margin-top: 15px;
            }}
            .server-item {{
                background: white;
                padding: 10px 15px;
                margin: 5px 0;
                border-radius: 4px;
                border-left: 3px solid #2196F3;
            }}
            .conversation {{
                padding: 20px;
            }}
            .message {{
                margin: 20px 0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .message-header {{
                padding: 15px 20px;
                font-weight: bold;
                font-size: 1.1em;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}
            .message-content {{
                padding: 20px;
                line-height: 1.6;
            }}
            .user-message {{
                border-left: 4px solid #2196F3;
            }}
            .user-message .message-header {{
                background: #e3f2fd;
                color: #1976d2;
            }}
            .assistant-message {{
                border-left: 4px solid #9c27b0;
            }}
            .assistant-message .message-header {{
                background: #f3e5f5;
                color: #7b1fa2;
            }}
            .content-block {{
                margin: 15px 0;
                padding: 15px;
                border-radius: 6px;
                border-left: 3px solid #dee2e6;
            }}
            .thinking-block {{
                background: #fff3e0;
                border-left-color: #ff9800;
            }}
            .thinking-header {{
                font-weight: bold;
                color: #f57c00;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
            }}
            .thinking-content {{
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                line-height: 1.4;
            }}
            .text-block {{
                background: #f8f9fa;
                border-left-color: #28a745;
            }}
            .text-content {{
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            .tool-use-block {{
                background: #e8f5e8;
                border-left-color: #4caf50;
            }}
            .tool-header {{
                font-weight: bold;
                color: #2e7d32;
                margin-bottom: 10px;
            }}
            .tool-details {{
                background: white;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }}
            .tool-result-block {{
                background: #f0f4f8;
                border-left-color: #607d8b;
            }}
            .tool-result-header {{
                font-weight: bold;
                color: #455a64;
                margin-bottom: 10px;
            }}
            .tool-result-content {{
                background: white;
                padding: 15px;
                border-radius: 4px;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                line-height: 1.4;
                max-height: 400px;
                overflow-y: auto;
            }}
            .redacted {{
                background: #ffebee;
                border-left-color: #f44336;
                color: #c62828;
                font-style: italic;
            }}
            .json-content {{
                background: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 0.85em;
                overflow-x: auto;
                border: 1px solid #e9ecef;
            }}
            .message-counter {{
                background: rgba(255,255,255,0.2);
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.9em;
            }}
            .summary-stats {{
                display: flex;
                justify-content: space-around;
                background: #e8f5e8;
                padding: 15px;
                margin-top: 20px;
                border-radius: 6px;
            }}
            .stat-item {{
                text-align: center;
            }}
            .stat-number {{
                font-size: 1.5em;
                font-weight: bold;
                color: #2e7d32;
            }}
            .stat-label {{
                color: #666;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 MCP Client Chat Log</h1>
                <p>Generated from: {log_filename}</p>
            </div>
            
            <div class="session-info">
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Session Start</div>
                        <div class="info-value">{session_start}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Session End</div>
                        <div class="info-value">{session_end}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Duration</div>
                        <div class="info-value">{duration_str}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Total Messages</div>
                        <div class="info-value">{total_messages}</div>
                    </div>
                </div>
                
                <div class="servers-section">
                    <div class="info-label">Connected Servers:</div>
    """
    
    # Add connected servers
    if connected_servers:
        for server_name, server_path in connected_servers.items():
            html_content += f'<div class="server-item"><strong>{server_name}:</strong> {server_path}</div>'
    else:
        html_content += '<div class="server-item">No servers connected</div>'
    
    html_content += """
                </div>
            </div>
            
            <div class="conversation">
    """
    
    # Process conversation history
    for i, message in enumerate(conversation_history):
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        
        # Determine message class
        message_class = 'user-message' if role == 'user' else 'assistant-message'
        
        html_content += f"""
            <div class="message {message_class}">
                <div class="message-header">
                    <span>👤 {role.title()}</span>
                    <span class="message-counter">Message #{i + 1}</span>
                </div>
                <div class="message-content">
        """
        
        # Process content based on type
        if isinstance(content, str):
            # Simple string content (user messages)
            formatted_text = _format_text_content(content)
            html_content += f'<div class="text-content">{formatted_text}</div>'
        elif isinstance(content, list):
            # Check if this is a user message with images
            has_images = any(isinstance(item, dict) and item.get('type') == 'image_url' for item in content)
            
            if has_images and role == 'user':
                # Process user message with images
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            # Text content
                            text = item.get('text', '')
                            formatted_text = _format_text_content(text)
                            html_content += f"""
                                <div class="content-block text-block">
                                    <div class="text-content">{formatted_text}</div>
                                </div>
                            """
                        elif item.get('type') == 'image_url':
                            # Image content
                            image_url = item.get('image_url', {}).get('url', '')
                            metadata = item.get('image_metadata', {})
                            
                            # Display image with metadata
                            html_content += '<div class="content-block" style="background: #f0f8ff; border-left-color: #2196F3;">'
                            html_content += '<div style="font-weight: bold; color: #1976d2; margin-bottom: 10px;">📸 Image</div>'
                            
                            if image_url:
                                # Display the base64 image
                                html_content += f'<img src="{image_url}" style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px 0;" alt="User uploaded image" />'
                            
                            # Display metadata if available
                            if metadata:
                                html_content += '<div style="font-size: 0.85em; color: #666; margin-top: 10px;">'
                                html_content += f'<strong>Filename:</strong> {_escape_html(metadata.get("filename", "unknown"))}<br>'
                                
                                orig_size = metadata.get('original_size', {})
                                if orig_size and orig_size.get('width', 0) > 0:
                                    html_content += f'<strong>Original size:</strong> {orig_size["width"]}×{orig_size["height"]}px<br>'
                                
                                resized_size = metadata.get('resized_size', {})
                                if resized_size and resized_size.get('width', 0) > 0:
                                    html_content += f'<strong>Displayed size:</strong> {resized_size["width"]}×{resized_size["height"]}px<br>'
                                
                                if 'file_size_kb' in metadata:
                                    html_content += f'<strong>File size:</strong> {metadata["file_size_kb"]} KB<br>'
                                
                                if 'note' in metadata:
                                    html_content += f'<em>{_escape_html(metadata["note"])}</em>'
                                
                                html_content += '</div>'
                            
                            html_content += '</div>'
            else:
                # Regular list processing (for assistant messages)
                # Complex content with multiple blocks
                for item in content:
                    if isinstance(item, dict):
                        content_type = item.get('type', 'text')
                        
                        if content_type == 'thinking':
                            thinking_text = item.get('thinking', '')
                            signature = item.get('signature', '')
                            html_content += f"""
                                <div class="content-block thinking-block">
                                    <div class="thinking-header">🧠 AI Reasoning / Thinking Process</div>
                                    <div class="thinking-content">{_escape_html(thinking_text)}</div>
                                    {f'<div style="margin-top: 10px; font-size: 0.8em; color: #666;">Signature: {signature[:50]}...</div>' if signature else ''}
                                </div>
                            """
                        elif content_type == 'redacted_thinking':
                            html_content += f"""
                                <div class="content-block redacted">
                                    <div class="thinking-header">🧠 AI Reasoning / Thinking Process (Redacted)</div>
                                    <div>This thinking content was redacted for safety reasons.</div>
                                </div>
                            """
                        elif content_type == 'text':
                            text = item.get('text', '')
                            formatted_text = _format_text_content(text)
                            html_content += f"""
                                <div class="content-block text-block">
                                    <div class="text-content">{formatted_text}</div>
                                </div>
                            """
                        elif content_type == 'tool_use':
                            tool_id = item.get('id', '')
                            tool_name = item.get('name', '')
                            tool_input = item.get('input', {})
                            html_content += f"""
                                <div class="content-block tool-use-block">
                                    <div class="tool-header">🔧 Tool Use: {tool_name}</div>
                                    <div class="tool-details">
                                        <strong>ID:</strong> {tool_id}<br>
                                        <strong>Arguments:</strong>
                                        <div class="json-content">{_format_json(tool_input)}</div>
                                    </div>
                                </div>
                            """
                        elif content_type == 'tool_result':
                            tool_use_id = item.get('tool_use_id', '')
                            result_content = item.get('content', '')
                            is_error = item.get('is_error', False)
                            
                            # Format result content
                            if isinstance(result_content, list):
                                formatted_result = ""
                                for result_item in result_content:
                                    if isinstance(result_item, dict) and result_item.get('type') == 'text':
                                        formatted_result += _format_text_content(result_item.get('text', ''))
                                    else:
                                        formatted_result += _format_text_content(str(result_item))
                            else:
                                formatted_result = _format_text_content(str(result_content))
                            
                            error_class = " redacted" if is_error else ""
                            html_content += f"""
                                <div class="content-block tool-result-block{error_class}">
                                    <div class="tool-result-header">📋 Tool Result{' (Error)' if is_error else ''}</div>
                                    <div style="font-size: 0.9em; margin-bottom: 10px;">Tool Use ID: {tool_use_id}</div>
                                    <div class="tool-result-content">{formatted_result}</div>
                                </div>
                            """
                    else:
                        # Fallback for other content types
                        formatted_text = _format_text_content(str(item))
                        html_content += f'<div class="text-content">{formatted_text}</div>'
        else:
            # Fallback for unknown content format
            formatted_text = _format_text_content(str(content))
            html_content += f'<div class="text-content">{formatted_text}</div>'
        
        html_content += '</div></div>'
    
    # Add summary statistics
    user_messages = len([m for m in conversation_history if m.get('role') == 'user'])
    assistant_messages = len([m for m in conversation_history if m.get('role') == 'assistant'])
    # Use the tool_uses_count we extracted from organized_conversation earlier
    # (tool_use items may not be present in conversation_history since tool_calls aren't saved)
    tool_uses = tool_uses_count
    
    html_content += f"""
                <div class="summary-stats">
                    <div class="stat-item">
                        <div class="stat-number">{total_messages}</div>
                        <div class="stat-label">Total Messages</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{user_messages}</div>
                        <div class="stat-label">User Messages</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{assistant_messages}</div>
                        <div class="stat-label">Assistant Messages</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{tool_uses}</div>
                        <div class="stat-label">Tool Uses</div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

def _escape_html(text):
    """Helper function to escape HTML characters"""
    if not isinstance(text, str):
        text = str(text)
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#x27;'))

def _format_text_content(text):
    """
    Format text content for HTML display with proper line breaks and spacing.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # First escape HTML characters
    text = _escape_html(text)
    
    # Convert double newlines to paragraph breaks
    text = text.replace('\n\n', '</p><p>')
    
    # Convert single newlines to line breaks
    text = text.replace('\n', '<br>')
    
    # Preserve multiple spaces by converting them to non-breaking spaces
    import re
    text = re.sub(r'  +', lambda m: '&nbsp;' * len(m.group()), text)
    
    # Wrap in paragraph tags if not already wrapped
    if not text.startswith('<p>') and not text.startswith('</p>'):
        text = f'<p>{text}</p>'
    
    # Clean up any empty paragraphs or malformed tags
    text = re.sub(r'<p></p>', '', text)
    text = re.sub(r'<p><br>', '<p>', text)
    text = re.sub(r'<br></p>', '</p>', text)
    
    return text

def _format_json(obj):
    """Format JSON object for display"""
    try:
        return _escape_html(json.dumps(obj, indent=2, ensure_ascii=False))
    except:
        return _escape_html(str(obj))

def save_chat_log_html(log_file, output_file=None):
    """
    Convert client chat logs to HTML and save to file.
    
    Args:
        log_file: Path to the JSON log file
        output_file: Optional output file path. If None, creates .html version of log_file
        
    Returns:
        str: Path to the generated HTML file
    """
    if output_file is None:
        # Generate output filename based on input
        base_name = os.path.splitext(log_file)[0]
        output_file = f"{base_name}.html"
    
    # Generate HTML content
    html_content = chat_log_to_html(log_file)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Client chat log saved to: {output_file}")
    return output_file

