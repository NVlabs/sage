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


def chat_log_to_html_qwen3(log_file):
    """
    Convert Qwen3-VL client chat logs to HTML format with inline CSS styling.
    
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
            
        # Validate format
        if not isinstance(log_data, dict):
            return "<html><body><h1>Invalid Qwen3 log format</h1></body></html>"
        
        if 'organized_conversation' not in log_data:
            return "<html><body><h1>Invalid Qwen3 log format - missing organized_conversation</h1></body></html>"
            
    except Exception as e:
        return f"<html><body><h1>Error reading log file: {str(e)}</h1></body></html>"
    
    # Extract session metadata
    session_start = log_data.get('session_start', 'Unknown')
    session_end = log_data.get('session_end', 'Unknown')
    connected_servers = log_data.get('connected_servers', {})
    conversation_summary = log_data.get('conversation_summary', {})
    
    # Get conversation data
    organized_conversation = log_data.get('organized_conversation', [])
    
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
        <title>Qwen3-VL Chat Log</title>
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
                background: linear-gradient(135deg, #9c27b0 0%, #7b1fa2 100%);
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
                border-left: 4px solid #9c27b0;
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
            .tool-message {{
                border-left: 4px solid #4caf50;
            }}
            .tool-message .message-header {{
                background: #e8f5e9;
                color: #2e7d32;
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
            .image-block {{
                background: #f0f8ff;
                border-left-color: #2196F3;
            }}
            .image-header {{
                font-weight: bold;
                color: #1976d2;
                margin-bottom: 10px;
            }}
            .image-content {{
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin: 10px 0;
            }}
            .image-metadata {{
                font-size: 0.85em;
                color: #666;
                margin-top: 10px;
            }}
            .tool-call-block {{
                background: #e8f5e8;
                border-left-color: #4caf50;
            }}
            .tool-call-header {{
                font-weight: bold;
                color: #2e7d32;
                margin-bottom: 10px;
            }}
            .tool-call-content {{
                background: white;
                padding: 10px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                white-space: pre-wrap;
                word-wrap: break-word;
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
            .error {{
                background: #ffebee;
                border-left-color: #f44336;
                color: #c62828;
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
                background: #f3e5f5;
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
                color: #7b1fa2;
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
                <h1>🤖 Qwen3-VL Chat Log</h1>
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
                        <div class="info-label">Total Exchanges</div>
                        <div class="info-value">{len(organized_conversation)}</div>
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
    
    # Process organized conversation
    for i, item in enumerate(organized_conversation):
        item_type = item.get('type', 'unknown')
        timestamp = item.get('timestamp', '')
        
        if item_type == 'user_message':
            # User message
            html_content += f"""
            <div class="message user-message">
                <div class="message-header">
                    <span>👤 User</span>
                    <span class="message-counter">#{i + 1} - {timestamp}</span>
                </div>
                <div class="message-content">
            """
            
            content = item.get('content', '')
            if isinstance(content, list):
                # Multi-modal content (text + images)
                for content_item in content:
                    if isinstance(content_item, dict):
                        if content_item.get('type') == 'text':
                            text = content_item.get('text', '')
                            formatted_text = _format_text_content(text)
                            html_content += f"""
                                <div class="content-block text-block">
                                    <div class="text-content">{formatted_text}</div>
                                </div>
                            """
                        elif content_item.get('type') == 'image_url':
                            image_url = content_item.get('image_url', {}).get('url', '')
                            metadata = content_item.get('image_metadata', {})
                            
                            html_content += '<div class="content-block image-block">'
                            html_content += '<div class="image-header">📸 Image</div>'
                            
                            if image_url:
                                html_content += f'<img src="{image_url}" class="image-content" alt="User uploaded image" />'
                            
                            if metadata:
                                html_content += '<div class="image-metadata">'
                                html_content += f'<strong>Filename:</strong> {_escape_html(metadata.get("filename", "unknown"))}<br>'
                                
                                orig_size = metadata.get('original_size', {})
                                if orig_size and orig_size.get('width', 0) > 0:
                                    html_content += f'<strong>Original size:</strong> {orig_size["width"]}×{orig_size["height"]}px<br>'
                                
                                resized_size = metadata.get('resized_size', {})
                                if resized_size and resized_size.get('width', 0) > 0:
                                    html_content += f'<strong>Displayed size:</strong> {resized_size["width"]}×{resized_size["height"]}px<br>'
                                
                                if 'file_size_kb' in metadata:
                                    html_content += f'<strong>File size:</strong> {metadata["file_size_kb"]:.2f} KB<br>'
                                
                                if 'note' in metadata:
                                    html_content += f'<em>{_escape_html(metadata["note"])}</em>'
                                
                                html_content += '</div>'
                            
                            html_content += '</div>'
            else:
                # Simple string content
                formatted_text = _format_text_content(str(content))
                html_content += f'<div class="text-content">{formatted_text}</div>'
            
            html_content += '</div></div>'
            
        elif item_type == 'assistant_response':
            # Assistant message
            iteration = item.get('iteration', '')
            html_content += f"""
            <div class="message assistant-message">
                <div class="message-header">
                    <span>🤖 Qwen3-VL</span>
                    <span class="message-counter">#{i + 1} - Iteration {iteration} - {timestamp}</span>
                </div>
                <div class="message-content">
            """
            
            # Add reasoning if present
            reasoning = item.get('reasoning', '')
            if reasoning and reasoning.strip():
                html_content += f"""
                    <div class="content-block thinking-block">
                        <div class="thinking-header">🧠 Model Reasoning / Thinking Process</div>
                        <div class="thinking-content">{_escape_html(reasoning)}</div>
                    </div>
                """
            
            # Add content
            content = item.get('content', '')
            if content and content.strip():
                formatted_content = _format_text_content(content)
                html_content += f"""
                    <div class="content-block text-block">
                        <div class="text-content">{formatted_content}</div>
                    </div>
                """
            
            html_content += '</div></div>'
            
        elif item_type == 'tool_result':
            # Tool result
            tool_name = item.get('tool_name', 'unknown')
            server = item.get('server', '')
            is_error = item.get('error', False)
            error_class = ' error' if is_error else ''
            
            html_content += f"""
            <div class="message tool-message{error_class}">
                <div class="message-header">
                    <span>🔧 Tool Result: {tool_name}</span>
                    <span class="message-counter">#{i + 1} - {server} - {timestamp}</span>
                </div>
                <div class="message-content">
            """
            
            content = item.get('content', '')
            formatted_content = _format_text_content(str(content))
            
            html_content += f"""
                    <div class="content-block tool-result-block{error_class}">
                        <div class="tool-result-header">📋 Result{' (Error)' if is_error else ''}</div>
                        <div class="tool-result-content">{formatted_content}</div>
                    </div>
            """
            
            html_content += '</div></div>'
    
    # Add summary statistics
    user_messages = conversation_summary.get('user_messages', 0)
    assistant_messages = conversation_summary.get('assistant_messages', 0)
    function_calls = conversation_summary.get('function_calls', 0)
    reasoning_entries = conversation_summary.get('reasoning_entries', 0)
    user_images = conversation_summary.get('user_images', 0)
    
    html_content += f"""
                <div class="summary-stats">
                    <div class="stat-item">
                        <div class="stat-number">{user_messages}</div>
                        <div class="stat-label">User Messages</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{assistant_messages}</div>
                        <div class="stat-label">Assistant Messages</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{function_calls}</div>
                        <div class="stat-label">Tool Calls</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{reasoning_entries}</div>
                        <div class="stat-label">Reasoning Entries</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{user_images}</div>
                        <div class="stat-label">Images</div>
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


def save_chat_log_html_qwen3(log_file, output_file=None):
    """
    Convert Qwen3-VL client chat logs to HTML and save to file.
    
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
    html_content = chat_log_to_html_qwen3(log_file)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Qwen3-VL chat log saved to: {output_file}")
    return output_file


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        save_chat_log_html_qwen3(log_file, output_file)
    else:
        print("Usage: python log_qwen3.py <log_file.json> [output_file.html]")

