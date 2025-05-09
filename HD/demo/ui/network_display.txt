import socket

class NetworkInfoDisplay:
    """Handles network information display"""
    @staticmethod
    def display_network_info():
        print("\n--- Current Network Information ---")
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            
            print(f"Hostname: {hostname}")
            print(f"IP Address: {ip_address}")
            print("-" * 40)
            
            return {
                "hostname": hostname,
                "ip_address": ip_address
            }
        except Exception as e:
            ProgressIndicator.show_warning(f"Could not get network info: {str(e)}")
            return None