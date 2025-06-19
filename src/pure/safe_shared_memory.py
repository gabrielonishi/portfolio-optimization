from multiprocessing.shared_memory import SharedMemory


class SafeSharedMemory:
    def __init__(self, *args, unlink=False, **kwargs):
        self.shm = SharedMemory(*args, **kwargs)
        self.unlink = unlink

    def __enter__(self):
        return self.shm

    def __exit__(self, exc_type, exc_value, traceback):
        self.shm.close()
        if self.unlink:
            self.shm.unlink()
