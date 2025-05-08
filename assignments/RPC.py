Practical no-1
Design a distributed application using RPC for remote computation where client submits an integer value to the server and server calculates factorial and returns the result to the client program.
1st file:Factserver.py
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
class FactorialServer:
    def calculate_factorial(self, n):
        if n < 0:
            raise ValueError("Input must be a non-negative integer.")
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)
# Create server
with SimpleXMLRPCServer(('localhost', 8000),
                        requestHandler=RequestHandler) as server:
    server.register_introspection_functions()
    # Register the FactorialServer class
    server.register_instance(FactorialServer())
    print("FactorialServer is ready to accept requests.")
    # Run the server's main loop
    server.serve_forever()


2nd file: Factclient.py
import xmlrpc.client
# Create an XML-RPC client
with xmlrpc.client.ServerProxy("http://localhost:8000/RPC2") as proxy:
    try:
        # Replace 5 with the desired integer value
        input_value = 5
        result = proxy.calculate_factorial(input_value)
        print(f"Factorial of {input_value} is: {result}")
    except Exception as e:
        print(f"Error: {e}")
Execution Steps
In PyCharm
1)Run First Server will show following Output
/home/gurukul/PycharmProjects/pythonProject/.venv/bin/python /home/gurukul/PycharmProjects/pythonProject/.venv/server.py
FactorialServer is ready to accept requests.
2)Run then Client & will show following Output
(.venv) gurukul@gurukul-ThinkCentre-M73:~/PycharmProjects/pythonProject/.venv$ python client.py
Factorial of 5 is: 120
(.venv) gurukul@gurukul-ThinkCentre-M73:~/PycharmProjects/pythonProject/.venv$
And In Server it add this after fetching with client
 127.0.0.1 - - [10/Apr/2024 15:10:12] "POST /RPC2 HTTP/1.1" 200 -

1. Open Command Prompt:
On Windows: Press Win + R, type cmd, and press Enter.
On Linux/macOS: Open a terminal.
2. Navigate to the Script's Directory:
Use the cd command to change to the directory where your Python script is located. 
For example:

3. Run the Python Script:
Use the python command followed by the name of your Python script to run it.
 For example: 

First execute factserver.py on command prompt
