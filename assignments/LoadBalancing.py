import time
import random


class Server:
    def __init__(self, name):
        self.name = name
        self.active_connections = 0


    def handle_request(self):
        self.active_connections += 1


    def finish_request(self):
        if self.active_connections > 0:
            self.active_connections -= 1


    def __str__(self):
        return f"{self.name} (Active: {self.active_connections})"




class RoundRobinBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.index = 0


    def get_server(self):
        if not self.servers:
            raise ValueError("No servers available")
        server = self.servers[self.index]
        self.index = (self.index + 1) % len(self.servers)
        return server




class LeastConnectionsBalancer:
    def __init__(self, servers):
        self.servers = servers


    def get_server(self):
        if not self.servers:
            raise ValueError("No servers available")
        return min(self.servers, key=lambda server: server.active_connections)




def simulate_requests(load_balancer, num_requests):
    for i in range(1, num_requests + 1):
        try:
            server = load_balancer.get_server()
            server.handle_request()
            print(f"Request {i} assigned to {server}")
        except ValueError as e:
            print(f"Error: {e}")
        time.sleep(0.5)  # Simulate processing delay


    print("\nFinal server load:")
    for server in load_balancer.servers:
        print(server)


    # Simulate finishing requests
    print("\nReleasing requests:")
    for server in load_balancer.servers:
        while server.active_connections > 0:
            server.finish_request()
            print(f"{server.name} processed a request. Remaining: {server.active_connections}")
            time.sleep(0.3)




# Create servers
servers = [Server("Server A"), Server("Server B"), Server("Server C")]


# Choose a Load Balancing Strategy
print("Simulating Round Robin Load Balancing:\n")
round_robin_balancer = RoundRobinBalancer(servers)
simulate_requests(round_robin_balancer, 10)


# Reset server connections for next test
for server in servers:
    server.active_connections = 0


print("\nSimulating Least Connections Load Balancing:\n")
least_conn_balancer = LeastConnectionsBalancer(servers)
simulate_requests(least_conn_balancer, 10)
