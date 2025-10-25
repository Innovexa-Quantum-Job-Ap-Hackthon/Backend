# -----------------------------------------
# Jupyter Server Config for Local Development
# -----------------------------------------
c = get_config()

# -------------------------------
# Authentication (local dev only)
# -------------------------------
# Disables token and password security.
# WARNING: Only use this for local development.
c.ServerApp.token = ''
c.ServerApp.password = ''

# -------------------------------
# Networking / CORS
# -------------------------------
# Allows your frontend application to make API requests to the Jupyter server.
c.ServerApp.allow_origin = '*'
c.ServerApp.allow_credentials = True

# -------------------------------
# Content Security Policy (CSP) - FIXED FOR IFRAME EMBEDDING
# -------------------------------
# This configuration allows your frontend to embed Jupyter in an iframe
c.ServerApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': (
            "frame-ancestors 'self' http://localhost:5173 https://localhost:5173 "
            "http://127.0.0.1:5173 http://localhost:3000 https://localhost:3000 "
            "http://127.0.0.1:3000; "
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' http://localhost:5173; "
            "style-src 'self' 'unsafe-inline' http://localhost:5173; "
            "img-src 'self' data: https: http://localhost:5173; "
            "connect-src 'self' http://localhost:5173 https://localhost:5173 "
            "http://127.0.0.1:5173 http://localhost:8888 https://localhost:8888 "
            "ws://localhost:8888 wss://localhost:8888; "
            "font-src 'self' data:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )
    }
}

# Alternative method using csp_directives (if tornado_settings doesn't work)
c.ServerApp.disable_check_xsrf = True  # Disable XSRF for development
c.ServerApp.allow_remote_access = True

# -------------------------------
# Kernel Management
# -------------------------------
# Automatically shut down idle kernels to save resources.
c.MappingKernelManager.cull_idle_timeout = 3600    # Cull kernels idle for 1 hour.
c.MappingKernelManager.cull_interval = 300          # Check for idle kernels every 5 minutes.
c.MappingKernelManager.cull_connected = False       # Do not cull if a client is connected.
c.MappingKernelManager.cull_busy = False            # Do not cull if the kernel is busy.
c.MappingKernelManager.kernel_info_timeout = 60

c.KernelManager.shutdown_wait_time = 5.0            # Wait 5 seconds before force-killing a kernel.

# -------------------------------
# Session Management
# -------------------------------
c.SessionManager.kernel_manager_class = 'jupyter_server.services.kernels.kernelmanager.MappingKernelManager'

# -------------------------------
# Notes
# -------------------------------
# This configuration is intended for local development only.
# Do not expose a server with this configuration to the public internet.
# For production, secure your server with authentication (tokens/passwords)
# or use a multi-user environment like JupyterHub.
