# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Class to closely monitor the status of the worlds and clean up the broken worlds.
"""

import asyncio
import logging
import os
import signal
import threading
import time
from asyncio import Queue as ASyncQ
from queue import Empty
from queue import Queue as SyncQ

from torch.distributed import DistNetworkError, DistStoreError

from multiworld.threadsafe_async import run_async

UPDATE_PERIOD = 0.3  # 300 ms
UPDATES_PER_CHECK = 10  # check every 3 sec
NOTICE_WAIT_TIMEOUT = 5  # 5 seconds

DEADLOCK_CHECK_WAIT_TIME = 5  # 5 sec
DEADLOCK_CHECK_ITERATIONS = 10  # 10 times

logger = logging.getLogger(__name__)

_deadlock_check_var = 0


class WorldStatus:
    """A class to store World's status."""

    def __init__(self):
        """Initialize an instance."""
        self.tick = 0
        self.broken = False


class WatchDog:
    """WatchDog class."""

    def __init__(self, event_q: SyncQ, action_q: ASyncQ):
        """Initialize a class instance."""
        self._event_q = event_q  # queue to receive "add world" event
        self._action_q = action_q

        self._loop = asyncio.get_running_loop()

        self._myworlds = dict()

        self._deadlock_check_trigger = threading.Event()

        threading.Thread(target=self._deadlock_check_thread, daemon=True).start()
        threading.Thread(target=self._monitor_thread, daemon=True).start()

    def _deadlock_check(self):
        global _deadlock_check_var

        logger.debug("let's check if main thread is blocked or not")

        os.kill(os.getpid(), signal.SIGUSR1)
        logger.debug("sent SIGUSR1")

        time.sleep(DEADLOCK_CHECK_WAIT_TIME)
        if _deadlock_check_var == 0:
            # Reaching here means that the main thread didn't handle
            # SIGUSR1 interrupt, which indicate the main thead is
            # blocked. So, there is nothing we can do.
            # so, we terminate the process.
            # TODO: graceful termination (saving some necessary states)
            #       need to think about what would be those states
            logger.debug("deadlock check failed; main thread is blocked")
            os.kill(os.getpid(), signal.SIGKILL)

        # reset the variable
        _deadlock_check_var = 0

    def _deadlock_check_thread(self):
        while True:
            self._deadlock_check_trigger.wait()
            self._deadlock_check_trigger.clear()

            # dist.init_process_group() get failed if an interrupt signal
            # is sent during the call. So, during the call, deadlock
            # check should be suspended. Currently, we do deadlock check
            # for a certain duration only in case a world gets broken.
            # DEADLOCK_CHECK_WAIT_TIME * DEADLOCK_CHECK_ITERATIONS = 50 seconds
            for _ in range(DEADLOCK_CHECK_ITERATIONS):
                self._deadlock_check()

    def _monitor_thread(self):
        tick = 0
        while True:
            empty = False
            try:
                store, world, rank, size = self._event_q.get_nowait()
            except Empty:
                empty = True

            if not empty:
                logger.debug(f"name: {world}, rank: {rank}, world size: {size}")
                self._myworlds[world] = (
                    store,
                    rank,
                    [WorldStatus() for i in range(size)],
                )

            # update tick for all the worlds that I belongs to
            broken_worlds = set()
            for world, value in self._myworlds.items():
                (store, rank, _) = value
                # increment tick by one
                try:
                    store.add(f"watchdog/{world}/{rank}", 1)
                except DistNetworkError as e:
                    logger.debug(f"world {world} is broken during add: {e}")
                    broken_worlds.add(world)

            cleanup_entries = (
                self._do_check() if tick % UPDATES_PER_CHECK == 0 else set()
            )

            cleanup_entries = cleanup_entries | broken_worlds
            for world in cleanup_entries:
                logger.debug(f"world {world} is broken")
                # remove the world from self._myworlds
                self._myworlds[world][2].clear()
                del self._myworlds[world]
                logger.debug(f"inform world {world} is broken")
                _, success = run_async(
                    self._action_q.put(world), self._loop, NOTICE_WAIT_TIMEOUT
                )
                if not success:
                    logger.debug(f"failed to inform the broken world {world}")
                    os.kill(os.getpid(), signal.SIGKILL)

            # if there is a broken world, check if deadlock occurs.
            if len(cleanup_entries) > 0:
                self._deadlock_check_trigger.set()

            tick += 1
            time.sleep(UPDATE_PERIOD)

    def _do_check(self) -> set[str]:
        # check the liveness of workers across worlds
        cleanup_entries = set()
        for world, value in self._myworlds.items():
            (store, my_rank, world_status_array) = value
            for rank in range(len(world_status_array)):
                if my_rank == rank:
                    # no need to check myself
                    continue

                try:
                    tick = int(store.get(f"watchdog/{world}/{rank}"))
                except DistNetworkError as e:
                    logger.debug(f"world {world} is broken during get: {e}")
                    cleanup_entries.add(world)
                    break
                except DistStoreError as e:
                    logger.debug(f"world {world} is broken during get: {e}")
                    cleanup_entries.add(world)
                    break

                if world_status_array[rank].tick == tick:
                    cleanup_entries.add(world)
                    break

                # update tick
                world_status_array[rank].tick = tick

        return cleanup_entries


def usr1_handler(signum, frame):
    """Handle SIGUSR1 signal.

    If this method is executed, it means the main thread is not blocked,
    meaning that it is not in a deadlock state. We set _deadlock_check_var
    to inform watchdog that the main thread is working fine.
    """
    global _deadlock_check_var
    logger.debug("received SIGUSR1")
    _deadlock_check_var = 1


# register usr1_handler
signal.signal(signal.SIGUSR1, usr1_handler)
