"""Class to manage communication between different worlds."""
from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed.world_manager import WorldManager


logger = logging.getLogger(__name__)


class CommunicationType:
    SEND = 1
    RECV = 2


class WorldCommunicator:
    """
    Class to manage communication between different worlds.

    NOTE: If using WorldCommunicationManager, use the API provided
    by the WorldCommunicator to create and manage worlds along with
    their communication links. Do not use the WorldManager API directly.
    """

    def __init__(self, world_manager: WorldManager):
        """Initialize a class instance."""
        self._world_manager = world_manager
        self._communication_threads = {}
        self._communication_commands = {}

        self._tensor_rx_q = queue.Queue()

    def add_world(self, world_name):
        """Add a new world to the world comm manager."""
        self._communication_commands[world_name] = queue.Queue()

        # NOTE(pranav): Might want to create separate threads for sending and receiving
        self._communication_threads[world_name] = threading.Thread(
            target=self._communication_thread, args=(world_name,)
        )
        self._communication_threads[world_name].start()

    def remove_world(self, world_name):
        """Remove a world from the world comm manager."""
        del self._communication_commands[world_name]
        del self._communication_threads[world_name]

    def _communication_thread(self, world_name):
        """Thread function to manage communication between worlds."""
        logger.debug(f"starting communication thread for {world_name}")

        while True:
            # This call blocks indefinitely until a command is received
            command, request_obj, tensor = self._communication_commands[
                world_name
            ].get()

            if command == CommunicationType.SEND:
                request_obj.wait()
            elif command == CommunicationType.RECV:
                request_obj.wait()
                self._tensor_rx_q.put(tensor)

    def send(self, tensor, world_name, rank):
        """Send a tensor to a specific rank in a world."""
        self._world_manager.set_world(world_name)

        # Catch any errors due to worker failures
        try:
            request_obj = dist.isend(tensor, dst=rank)

            self._communication_commands[world_name].put(
                (CommunicationType.SEND, request_obj, tensor)
            )

            return request_obj
        except RuntimeError as e:
            error_message = str(e)

            if "Connection closed by peer" in error_message:
                logger.warn("Ignoring Connection closed by peer error")
            elif "Connection reset by peer" in error_message:
                logger.warn("Ignoring Connection reset by peer error")
            else:
                raise e

        return None

    def recv(self, tensor, world_name, rank):
        """Receive a tensor from a specific rank in a world."""
        self._world_manager.set_world(world_name)

        # Catch any errors due to worker failures
        try:
            request_obj = dist.irecv(tensor, src=rank)

            self._communication_commands[world_name].put(
                (CommunicationType.RECV, request_obj, tensor)
            )

            return request_obj
        except RuntimeError as e:
            error_message = str(e)

            if "Connection closed by peer" in error_message:
                logger.warn("Ignoring Connection closed by peer error")
            elif "Connection reset by peer" in error_message:
                logger.warn("Ignoring Connection reset by peer error")
            else:
                raise e

        return None

    @property
    def rx_q(self):
        """Return the rx queue for received tensors."""
        return self._tensor_rx_q
