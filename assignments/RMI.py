Server Implementation:
 # server.py
import Pyro4
@Pyro4.expose
class StringConcatenationServer:
    def concatenate_strings(self, str1, str2):
        result = str1 + str2
        return result


def main():
    daemon = Pyro4.Daemon()  # Create a Pyro daemon
    ns = Pyro4.locateNS()  # Locate the Pyro nameserver


    # Create an instance of the server class
    server = StringConcatenationServer()


    # Register the server object with the Pyro nameserver
    uri = daemon.register(server)
    ns.register("string.concatenation", uri)


    print("Server URI:", uri)


    with open("server_uri.txt", "w") as f:
        f.write(str(uri))


    daemon.requestLoop()


if __name__ == "__main__":
    main()

Client Implementation:
# client.py
import Pyro4


def main():
    with open("server_uri.txt", "r") as f:
        uri = f.read()


    server = Pyro4.Proxy(uri)  # Connect to the remote server


    str1 = input("Enter the first string: ")
    str2 = input("Enter the second string: ")


    result = server.concatenate_strings(str1, str2)


    print("Concatenated Result:", result)


if __name__ == "__main__":
    main()

Steps to Run:
Install Pyro4 library
Then Use command Pyro4-ns
And use following steps
1)Save the server code in a file, e.g., server.py
2)Save the client code in a file, e.g., client.py
3)Open a terminal and run the server: python server.py
4)you will get server uri paste it inserver_uri.txt file.keep it in same folder where you have stored python files.
5)Open another terminal and run the client: python client.py
6)enter the values for  concatenation.
This example demonstrates a basic setup for using Pyro4 to create a distributed application for string concatenation. Adjust the strings in the client code as needed. Note that this is a simple example, and in a real-world scenario, you might want to handle exceptions, error checking, and security considerations.
Reference Link: https://www.javatpoint.com/RMI
