hf auth login --token HF_TOKEN
pip install diffusers -U
pip install transformers -U
pip install accelerate -U
pip install sentencepiece -U
pip install pillow -U

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
pip install flask werkzeug psutil flask-cors requests
echo "import os
import io
import tempfile
import time
import psutil
import json
import sys
import argparse
import threading
from flask import Flask, request, jsonify, send_file, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
import random
import socket
import subprocess
import torch
from PIL import Image

# Parse command line arguments
parser = argparse.ArgumentParser(description='Flux Image Generation Server')
parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use (for logging purposes)')
args = parser.parse_args()

# Log which GPU is being used (CUDA_VISIBLE_DEVICES handles the actual GPU selection)
if args.gpu is not None:
    print(f'Running on GPU {args.gpu} (via CUDA_VISIBLE_DEVICES)')
else:
    print('Using default GPU configuration')

from diffusers import FluxPipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Metrics tracking
start_time = time.time()
request_count = 0
error_count = 0
total_generation_time = 0.0

# Load the pipeline once when the server starts
print('Loading Flux pipeline...')
pipeline = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-Krea-dev', torch_dtype=torch.bfloat16)
pipeline.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
print('Pipeline loaded successfully!')

def get_network_ip():
    '''Get the network IP address of this machine.'''
    try:
        # Try to get the IP address by connecting to a public DNS
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except:
        try:
            # Fallback to hostname -I
            ip_address = subprocess.check_output(['hostname', '-I'], text=True).strip().split()[0]
            return ip_address
        except:
            return '0.0.0.0'

@app.route('/', methods=['GET'])
def root():
    network_ip = get_network_ip()
    port = args.port
    return jsonify({
        'service': 'Flux Image Generation API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': f'http://{network_ip}:{port}/health',
            'metrics': f'http://{network_ip}:{port}/metrics', 
            'openapi': f'http://{network_ip}:{port}/openapi.json',
            'models': f'http://{network_ip}:{port}/api/v1/models',
            'generate': f'http://{network_ip}:{port}/generate'
        },
        'documentation': f'http://{network_ip}:{port}/openapi.json'
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'gpu_available': True})

@app.route('/metrics', methods=['GET'])
def metrics():
    global request_count, error_count, total_generation_time, start_time
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # GPU metrics (if available)
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                'gpu_count': torch.cuda.device_count(),
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
                'gpu_name': torch.cuda.get_device_name(0)
            }
    except:
        pass
    
    # Application metrics
    uptime = time.time() - start_time
    avg_generation_time = total_generation_time / max(1, request_count) if request_count > 0 else 0
    
    metrics_data = {
        'system': {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / 1024**3,
            'memory_total_gb': memory.total / 1024**3,
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / 1024**3,
            'disk_total_gb': disk.total / 1024**3
        },
        'gpu': gpu_info,
        'application': {
            'uptime_seconds': uptime,
            'total_requests': request_count,
            'total_errors': error_count,
            'total_generation_time_seconds': total_generation_time,
            'average_generation_time_seconds': avg_generation_time,
            'error_rate_percent': (error_count / max(1, request_count)) * 100
        },
        'timestamp': time.time()
    }
    
    return jsonify(metrics_data)

@app.route('/openapi.json', methods=['GET'])
def openapi_spec():
    response = jsonify(get_openapi_spec())
    response.headers['Content-Type'] = 'application/json'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

def get_openapi_spec():
    network_ip = get_network_ip()
    port = args.port
    spec = {
        'openapi': '3.0.0',
        'info': {
            'title': 'Flux Image Generation API',
            'version': '1.0.0',
            'description': 'API for generating images using Flux model',
            'contact': {
                'name': 'Flux API Support'
            }
        },
        'servers': [
            {
                'url': f'http://{network_ip}:{port}',
                'description': 'Flux Image Generation Server'
            }
        ],
        'paths': {
            '/health': {
                'get': {
                    'summary': 'Health check endpoint',
                    'responses': {
                        '200': {
                            'description': 'Server is healthy',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'object',
                                        'properties': {
                                            'status': {'type': 'string'},
                                            'gpu_available': {'type': 'boolean'}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            '/metrics': {
                'get': {
                    'summary': 'System and application metrics',
                    'responses': {
                        '200': {
                            'description': 'Metrics data',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'object',
                                        'properties': {
                                            'system': {'type': 'object'},
                                            'gpu': {'type': 'object'},
                                            'application': {'type': 'object'},
                                            'timestamp': {'type': 'number'}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            '/api/v1/models': {
                'get': {
                    'summary': 'List available models',
                    'responses': {
                        '200': {
                            'description': 'List of available models',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'object',
                                        'properties': {
                                            'data': {
                                                'type': 'array',
                                                'items': {
                                                    'type': 'object',
                                                    'properties': {
                                                        'id': {'type': 'string'},
                                                        'object': {'type': 'string'},
                                                        'created': {'type': 'integer'},
                                                        'owned_by': {'type': 'string'}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            '/generate': {
                'post': {
                    'summary': 'Generate image from text',
                    'requestBody': {
                        'required': True,
                        'content': {
                            'application/json': {
                                'schema': {
                                    'type': 'object',
                                    'properties': {
                                        'prompt': {
                                            'type': 'string',
                                            'description': 'Text description of the image to generate'
                                        },
                                        'height': {
                                            'type': 'integer',
                                            'description': 'Image height (default: 1024)',
                                            'default': 1024
                                        },
                                        'width': {
                                            'type': 'integer',
                                            'description': 'Image width (default: 1024)',
                                            'default': 1024
                                        },
                                        'guidance_scale': {
                                            'type': 'number',
                                            'description': 'Guidance scale (default: 4.5)',
                                            'default': 4.5
                                        },
                                        'seed': {
                                            'type': 'integer',
                                            'description': 'Random seed for generation (optional)'
                                        }
                                    },
                                    'required': ['prompt']
                                }
                            }
                        }
                    },
                    'responses': {
                        '200': {
                            'description': 'Generated image in PNG format',
                            'content': {
                                'image/png': {
                                    'schema': {
                                        'type': 'string',
                                        'format': 'binary'
                                    }
                                }
                            }
                        },
                        '500': {
                            'description': 'Generation error',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'object',
                                        'properties': {
                                            'error': {'type': 'string'}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return spec

@app.route('/api/v1/models', methods=['GET'])
def list_models():
    models_data = {
        'object': 'list',
        'data': [
            {
                'id': 'black-forest-labs/FLUX.1-Krea-dev',
                'object': 'model',
                'created': int(start_time),
                'owned_by': 'black-forest-labs',
                'description': 'Flux text-to-image generation model',
                'capabilities': ['text-to-image', 'image-generation'],
                'max_input_length': 512,
                'supported_formats': ['png']
            }
        ]
    }
    return jsonify(models_data)

@app.route('/docs', methods=['GET'])
def api_docs():
    # Return a simple HTML page pointing to the OpenAPI spec
    network_ip = get_network_ip()
    port = args.port
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flux API Documentation</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .endpoint {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            code {{ background: #e0e0e0; padding: 2px 4px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <h1>🚀 Flux Image Generation API</h1>
        <p>Welcome to the Flux API for text-to-image generation!</p>
        
        <h2>📋 Available Endpoints:</h2>
        <div class="endpoint">
            <strong>OpenAPI Specification:</strong><br>
            <code>GET <a href="http://{network_ip}:{port}/openapi.json">http://{network_ip}:{port}/openapi.json</a></code>
        </div>
        <div class="endpoint">
            <strong>Health Check:</strong><br>
            <code>GET <a href="http://{network_ip}:{port}/health">http://{network_ip}:{port}/health</a></code>
        </div>
        <div class="endpoint">
            <strong>System Metrics:</strong><br>
            <code>GET <a href="http://{network_ip}:{port}/metrics">http://{network_ip}:{port}/metrics</a></code>
        </div>
        <div class="endpoint">
            <strong>Available Models:</strong><br>
            <code>GET <a href="http://{network_ip}:{port}/api/v1/models">http://{network_ip}:{port}/api/v1/models</a></code>
        </div>
        <div class="endpoint">
            <strong>Generate Image:</strong><br>
            <code>POST http://{network_ip}:{port}/generate</code><br>
            <em>Body: {{"prompt": "your description", "height": 1024, "width": 1024, "guidance_scale": 4.5}}</em>
        </div>
        
        <h2>🔧 Testing the API:</h2>
        <p>You can test the API using curl:</p>
        <pre><code>curl -X POST http://{network_ip}:{port}/generate \\
  -H "Content-Type: application/json" \\
  -d '{{"prompt": "a red sports car"}}' \\
  --output image.png</code></pre>
    </body>
    </html>
    '''
    return html_content, 200, {{'Content-Type': 'text/html'}}

@app.route('/swagger', methods=['GET'])
def swagger_redirect():
    # Redirect to docs page
    return redirect('/docs')

@app.route('/api', methods=['GET'])
def api_info():
    # Alternative API info endpoint
    network_ip = get_network_ip()
    port = args.port
    return jsonify({{
        'api_version': '1.0.0',
        'service': 'Flux',
        'openapi_url': f'http://{{network_ip}}:{{port}}/openapi.json',
        'docs_url': f'http://{{network_ip}}:{{port}}/docs',
        'models_url': f'http://{{network_ip}}:{{port}}/api/v1/models'
    }})

@app.route('/openapi.json/test', methods=['GET'])
def test_openapi():
    # Test endpoint to validate OpenAPI spec
    try:
        spec = get_openapi_spec()
        return jsonify({{
            'valid': True,
            'spec_keys': list(spec.keys()),
            'paths_count': len(spec.get('paths', {{}})),
            'spec_size': len(str(spec))
        }})
    except Exception as e:
        return jsonify({{
            'valid': False,
            'error': str(e)
        }}), 500

# Job tracking for worker servers
worker_jobs = {}
worker_jobs_lock = threading.Lock()

@app.route('/generate', methods=['POST'])
def generate_image():
    global request_count, error_count, total_generation_time
    
    request_count += 1
    
    try:
        data = request.get_json()
        prompt = data.get('prompt', 'A simple image')
        height = data.get('height', 1024)
        width = data.get('width', 1024)
        guidance_scale = data.get('guidance_scale', 4.5)
        seed = data.get('seed', random.randint(1, 1000000))
        job_id = f'{int(time.time()*1000)}_{seed}'
        
        # Immediately acknowledge receipt
        with worker_jobs_lock:
            worker_jobs[job_id] = {
                'status': 'processing',
                'prompt': prompt,
                'seed': seed,
                'height': height,
                'width': width,
                'guidance_scale': guidance_scale,
                'created_at': time.time()
            }
        
        print(f'Job {job_id}: Acknowledged request for: {prompt}')
        
        # Start generation in background thread
        def generate_async():
            global total_generation_time, error_count
            generation_start = time.time()
            
            try:
                print(f'Job {job_id}: Starting generation...')
                
                # Set seed for reproducibility
                import torch
                generator = torch.Generator().manual_seed(seed)
                
                # Run the pipeline
                image = pipeline(
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images[0]
                
                # Save to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                file_content = img_byte_arr.getvalue()
                
                # Track successful generation time
                generation_end = time.time()
                total_generation_time += (generation_end - generation_start)
                
                # Update job status
                with worker_jobs_lock:
                    worker_jobs[job_id] = {
                        'status': 'completed',
                        'prompt': prompt,
                        'seed': seed,
                        'height': height,
                        'width': width,
                        'guidance_scale': guidance_scale,
                        'created_at': worker_jobs[job_id]['created_at'],
                        'completed_at': time.time(),
                        'file_content': file_content,
                        'generation_time': generation_end - generation_start
                    }
                
                print(f'Job {job_id}: Generation completed successfully')
                
            except Exception as e:
                error_count += 1
                print(f'Job {job_id}: Generation failed: {e}')
                with worker_jobs_lock:
                    worker_jobs[job_id] = {
                        'status': 'failed',
                        'prompt': prompt,
                        'seed': seed,
                        'created_at': worker_jobs[job_id]['created_at'],
                        'error': str(e)
                    }
        
        # Start async generation
        threading.Thread(target=generate_async, daemon=True).start()
        
        # Return acknowledgment immediately
        return jsonify({
            'status': 'accepted',
            'job_id': job_id,
            'message': 'Request received and processing started'
        }), 202
        
    except Exception as e:
        error_count += 1
        return jsonify({'error': str(e)}), 500

@app.route('/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    with worker_jobs_lock:
        job = worker_jobs.get(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job['status'] == 'completed':
        # Return the file
        file_content = job['file_content']
        job_seed = job['seed']
        return send_file(
            io.BytesIO(file_content),
            as_attachment=True,
            download_name=f'generated_image_{job_seed}.png',
            mimetype='image/png'
        )
    elif job['status'] == 'failed':
        return jsonify({
            'status': 'failed',
            'error': job.get('error', 'Unknown error')
        }), 500
    else:
        return jsonify({
            'status': job['status'],
            'job_id': job_id,
            'message': 'Job is still processing'
        }), 202

if __name__ == '__main__':
    port = args.port
    host = '0.0.0.0'  # Allow access from network
    
    # Get network IP for display
    network_ip = get_network_ip()
    
    # Write network address to file for external access
    address_info = {
        'network_url': f'http://{network_ip}:{port}',
        'local_url': f'http://localhost:{port}',
        'host': network_ip,
        'port': port,
        'timestamp': str(subprocess.check_output(['date'], text=True).strip()),
        'endpoints': {
            'root': f'http://{network_ip}:{port}/',
            'health': f'http://{network_ip}:{port}/health',
            'metrics': f'http://{network_ip}:{port}/metrics',
            'docs': f'http://{network_ip}:{port}/docs',
            'openapi': f'http://{network_ip}:{port}/openapi.json',
            'openapi_test': f'http://{network_ip}:{port}/openapi.json/test',
            'api_info': f'http://{network_ip}:{port}/api',
            'models': f'http://{network_ip}:{port}/api/v1/models',
            'generate': f'http://{network_ip}:{port}/generate'
        }
    }
    
    # Save to JSON file

    print('=' * 60)
    print('🚀 Flux Image Generation Server')
    print('=' * 60)
    print(f'🌐 Server will be available at:')
    print(f'   Local: http://localhost:{port}')
    print(f'   Network: http://{network_ip}:{port}')
    print('=' * 60)
    print('📋 API Endpoints:')
    print(f'   Root/API Info: http://{network_ip}:{port}/')
    print(f'   Health Check:  http://{network_ip}:{port}/health')
    print(f'   Metrics:       http://{network_ip}:{port}/metrics')
    print(f'   📖 Docs Page:    http://{network_ip}:{port}/docs')
    print(f'   📄 OpenAPI Spec: http://{network_ip}:{port}/openapi.json')
    print(f'   🧪 Test OpenAPI: http://{network_ip}:{port}/openapi.json/test')
    print(f'   📝 Models List:  http://{network_ip}:{port}/api/v1/models')
    print(f'   🎯 Generate Image:  http://{network_ip}:{port}/generate')
    print('=' * 60)
    print('🔗 Share the Network URL with others on the same network!')
    print('⚠️  Make sure firewall allows connections on this port')
    print('=' * 60)
    print('🔍 Troubleshooting OpenAPI:')
    print(f'   Test OpenAPI validity: http://{network_ip}:{port}/openapi.json/test')
    print(f'   Direct OpenAPI access: curl http://{network_ip}:{port}/openapi.json')
    print('=' * 60)
    
    # Run Flask server
    app.run(host=host, port=port, debug=False)" > server.py

# Create central server that distributes requests
echo "import os
import json
import time
import random
import requests
import subprocess
import threading
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import socket

app = Flask(__name__)
CORS(app)

# Track worker servers
worker_servers = []
current_worker = 0
worker_lock = threading.Lock()

# Track central server jobs
central_jobs = {}
central_jobs_lock = threading.Lock()

def get_network_ip():
    '''Get the network IP address of this machine.'''
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except:
        try:
            ip_address = subprocess.check_output(['hostname', '-I'], text=True).strip().split()[0]
            return ip_address
        except:
            return '0.0.0.0'

def get_next_worker():
    '''Round-robin load balancing with thread safety'''
    global current_worker
    if not worker_servers:
        return None
    
    with worker_lock:
        worker = worker_servers[current_worker % len(worker_servers)]
        current_worker += 1
        return worker

def check_worker_health(worker_url):
    '''Check if a worker server is healthy'''
    try:
        response = requests.get(f'{worker_url}/health', timeout=5)
        return response.status_code == 200
    except:
        return False

@app.route('/', methods=['GET'])
def root():
    network_ip = get_network_ip()
    port = 8080
    return jsonify({
        'service': 'Flux Image Generation API (Central Distributor)',
        'version': '1.0.0',
        'status': 'running',
        'worker_servers': len(worker_servers),
        'endpoints': {
            'health': f'http://{network_ip}:{port}/health',
            'metrics': f'http://{network_ip}:{port}/metrics', 
            'openapi': f'http://{network_ip}:{port}/openapi.json',
            'models': f'http://{network_ip}:{port}/api/v1/models',
            'generate': f'http://{network_ip}:{port}/generate'
        },
        'documentation': f'http://{network_ip}:{port}/openapi.json'
    })

@app.route('/health', methods=['GET'])
def health_check():
    healthy_workers = sum(1 for worker in worker_servers if check_worker_health(worker))
    return jsonify({
        'status': 'healthy' if healthy_workers > 0 else 'degraded',
        'total_workers': len(worker_servers),
        'healthy_workers': healthy_workers,
        'gpu_available': healthy_workers > 0
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    # Aggregate metrics from all workers
    total_metrics = {
        'workers': len(worker_servers),
        'healthy_workers': 0,
        'worker_metrics': []
    }
    
    for worker_url in worker_servers:
        try:
            response = requests.get(f'{worker_url}/metrics', timeout=5)
            if response.status_code == 200:
                worker_data = response.json()
                total_metrics['worker_metrics'].append({
                    'url': worker_url,
                    'metrics': worker_data,
                    'status': 'healthy'
                })
                total_metrics['healthy_workers'] += 1
            else:
                total_metrics['worker_metrics'].append({
                    'url': worker_url,
                    'status': 'unhealthy'
                })
        except:
            total_metrics['worker_metrics'].append({
                'url': worker_url,
                'status': 'unreachable'
            })
    
    return jsonify(total_metrics)

@app.route('/openapi.json', methods=['GET'])
def openapi_spec():
    # Forward to a healthy worker
    worker = get_next_worker()
    if not worker:
        return jsonify({'error': 'No healthy workers available'}), 503
    
    try:
        response = requests.get(f'{worker}/openapi.json', timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({'error': 'Failed to get OpenAPI spec'}), 500
    except:
        return jsonify({'error': 'Worker unreachable'}), 503

@app.route('/api/v1/models', methods=['GET'])
def list_models():
    # Forward to a healthy worker
    worker = get_next_worker()
    if not worker:
        return jsonify({'error': 'No healthy workers available'}), 503
    
    try:
        response = requests.get(f'{worker}/api/v1/models', timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({'error': 'Failed to get models list'}), 500
    except:
        return jsonify({'error': 'Worker unreachable'}), 503

@app.route('/docs', methods=['GET'])
def api_docs():
    # Forward to a healthy worker
    worker = get_next_worker()
    if not worker:
        return 'No healthy workers available', 503
    
    try:
        response = requests.get(f'{worker}/docs', timeout=10)
        if response.status_code == 200:
            return response.text, 200, {'Content-Type': 'text/html'}
        else:
            return 'Failed to get documentation', 500
    except:
        return 'Worker unreachable', 503

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        job_id = f'central_{int(time.time()*1000)}_{random.randint(1000, 9999)}'
        
        # Immediately acknowledge receipt
        with central_jobs_lock:
            central_jobs[job_id] = {
                'status': 'processing',
                'request_data': data,
                'created_at': time.time()
            }
        
        print(f'Central Job {job_id}: Acknowledged request')
        
        # Start async processing
        def process_async():
            attempts = 0
            max_attempts = len(worker_servers) if worker_servers else 1
            
            while attempts < max_attempts:
                worker = get_next_worker()
                if not worker:
                    with central_jobs_lock:
                        central_jobs[job_id]['status'] = 'failed'
                        central_jobs[job_id]['error'] = 'No workers available'
                    return
                
                try:
                    print(f'Central Job {job_id}: Forwarding to worker {worker}')
                    
                    # Send request to worker
                    response = requests.post(
                        f'{worker}/generate',
                        json=data,
                        timeout=10
                    )
                    
                    if response.status_code == 202:
                        # Worker acknowledged, get job_id
                        worker_job_data = response.json()
                        worker_job_id = worker_job_data.get('job_id')
                        
                        print(f'Central Job {job_id}: Worker acknowledged with job {worker_job_id}')
                        
                        # Poll worker for completion
                        while True:
                            time.sleep(2)  # Poll every 2 seconds
                            
                            try:
                                status_response = requests.get(
                                    f'{worker}/job/{worker_job_id}',
                                    timeout=10
                                )
                                
                                if status_response.status_code == 200:
                                    # Job completed, save result
                                    print(f'Central Job {job_id}: Worker completed successfully')
                                    with central_jobs_lock:
                                        central_jobs[job_id]['status'] = 'completed'
                                        central_jobs[job_id]['file_content'] = status_response.content
                                        central_jobs[job_id]['completed_at'] = time.time()
                                    return
                                elif status_response.status_code == 500:
                                    # Job failed on worker
                                    error_data = status_response.json()
                                    error_message = error_data.get('error')
                                    print(f'Central Job {job_id}: Worker failed - {error_message}')
                                    with central_jobs_lock:
                                        central_jobs[job_id]['status'] = 'failed'
                                        central_jobs[job_id]['error'] = error_data.get('error', 'Worker generation failed')
                                    return
                                # else status_code == 202, keep polling
                                
                            except Exception as poll_error:
                                print(f'Central Job {job_id}: Polling error - {poll_error}')
                                attempts += 1
                                break
                    else:
                        # Worker failed to accept
                        attempts += 1
                        continue
                        
                except Exception as e:
                    attempts += 1
                    print(f'Central Job {job_id}: Worker {worker} failed: {e}')
                    continue
            
            # All workers failed
            with central_jobs_lock:
                central_jobs[job_id]['status'] = 'failed'
                central_jobs[job_id]['error'] = 'All workers failed or unavailable'
        
        # Start async processing
        threading.Thread(target=process_async, daemon=True).start()
        
        # Return acknowledgment immediately
        return jsonify({
            'status': 'accepted',
            'job_id': job_id,
            'message': 'Request received and processing started'
        }), 202
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    with central_jobs_lock:
        job = central_jobs.get(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job['status'] == 'completed':
        # Return the file
        file_content = job['file_content']
        return app.response_class(
            file_content,
            mimetype='image/png',
            headers={
                'Content-Disposition': 'attachment; filename=image.png'
            }
        )
    elif job['status'] == 'failed':
        return jsonify({
            'status': 'failed',
            'error': job.get('error', 'Unknown error')
        }), 500
    else:
        return jsonify({
            'status': job['status'],
            'job_id': job_id,
            'message': 'Job is still processing'
        }), 202

if __name__ == '__main__':
    import sys
    
    # Get worker server URLs from command line arguments
    if len(sys.argv) > 1:
        worker_servers = sys.argv[1].split(',')
        print(f'Worker servers: {worker_servers}')
    else:
        print('No worker servers specified')
    
    network_ip = get_network_ip()
    port = 8080
    
    print('=' * 60)
    print('🚀 Flux Central Distribution Server')
    print('=' * 60)
    print(f'🌐 Central server available at:')
    print(f'   Local: http://localhost:{port}')
    print(f'   Network: http://{network_ip}:{port}')
    print(f'📊 Managing {len(worker_servers)} worker servers')
    for i, worker in enumerate(worker_servers):
        print(f'   Worker {i+1}: {worker}')
    print('=' * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)
" > central_server.py

# Detect available GPUs
echo "Detecting available GPUs..."
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
echo "Found $GPU_COUNT GPUs"

if [ $GPU_COUNT -eq 0 ]; then
    echo "No GPUs available. Starting single CPU server..."
    python server.py --port 8080
    exit 0
fi

# Start worker servers on each GPU
WORKER_URLS=""
for ((i=0; i<$GPU_COUNT; i++)); do
    PORT=$((8081 + i))
    echo "Starting Flux worker server on GPU $i, port $PORT..."
    CUDA_VISIBLE_DEVICES=$i python server.py --port $PORT --gpu $i &
    
    # Build worker URLs list
    if [ $i -eq 0 ]; then
        WORKER_URLS="http://localhost:$PORT"
    else
        WORKER_URLS="$WORKER_URLS,http://localhost:$PORT"
    fi
    
    # Give each server some time to start
    sleep 5
done

echo "All Flux worker servers started. Worker URLs: $WORKER_URLS"

# Wait for all workers to be ready
echo "Waiting for workers to be ready..."
sleep 10

# Start central distributor server
echo "Starting Flux central distributor server on port 8080..."
python central_server.py "$WORKER_URLS"