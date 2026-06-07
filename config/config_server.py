"""
config file to store the ip and ports
Note: set SERVER_IP via the SERVER_IP environment variable before running.
"""
import os
SERVER_IP = os.environ.get('SERVER_IP', 'YOUR_SERVER_IP_HERE')
VISUAL_SERVER_PORT = 8080 # port number
DEPTH_SERVER_PORT = 4096