"""Class to manage communication between different worlds"""
from torch.distributed.world_manager import WorldManager
import torch.distributed as dist
import threading
import queue

logger = logging.getLogger(__name__)

class CommunicationType:
    SEND = 1
    RECV = 2

class WorldCommunicationManager:
    """
    Class to manage communication between different worlds

    NOTE: If using WorldCommunicationManager, use the API provided
    by the WorldCommunicationManager to create and manage worlds along with
    their communication links. Do not use the WorldManager API directly.
    """
    def __init__(self, world_manager: WorldManager):
        self._world_manager = world_manager
        self._communication_threads = {}
        self._communication_commands = {}

        self._received_tensors = queue.Queue()

    def add_world(self, world_name, world=None):
        """
        Add a new world to the world manager
        """
        self._world_manager.add_world(world_name, world)

        self._communication_commands[world_name] = queue.Queue()

        # NOTE(pranav): Might want to create separate threads for sending and receiving
        self._communication_threads[world_name] = threading.Thread(
            target=self._communication_thread, args=(world_name,))

    def remove_world(self, world_name):
        """
        Remove a world from the world manager
        """
        self._world_manager.remove_world(world_name)

        del self._communication_commands[world_name]
        del self._communication_threads[world_name]

    def _communication_thread(self, world_name):
        """
        Thread function to manage communication between worlds
        """
        while True:
            # This call blocks indefinitely until a command is received
            command, request_obj, tensor = self._communication_commands[world_name].get()

            if command == CommunicationType.SEND:
                request_obj.wait()
            elif command == CommunicationType.RECV:
                request_obj.wait()
                self._received_tensors.put(tensor)

    def send(self, tensor, world_name, rank):
        """
        Send a tensor to a specific rank in a world
        """
        self._world_manager.set_world(world_name)
        request_obj = dist.isend(tensor, dst=rank)

        self._communication_commands[world_name].put((CommunicationType.SEND, request_obj, tensor))

        return request_obj

    def recv(self, tensor, world_name, rank):
        """
        Receive a tensor from a specific rank in a world
        """
        self._world_manager.set_world(world_name)
        request_obj = dist.irecv(tensor, src=rank)

        self._communication_commands[world_name].put((CommunicationType.RECV, request_obj, tensor))

        return request_obj
