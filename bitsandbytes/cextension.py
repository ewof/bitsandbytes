import ctypes as ct
import os
import torch

from pathlib import Path
from warnings import warn

# from bitsandbytes.cuda_setup.main import CUDASetup



class CUDASetup(object):
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def generate_instructions(self):
        self.add_log_entry('CUDA SETUP: Something unexpected happened. Please compile from source:')
        self.add_log_entry('git clone git@github.com:TimDettmers/bitsandbytes.git')
        self.add_log_entry('cd bitsandbytes')
        self.add_log_entry("<make_cmd here, commented out>")
        self.add_log_entry('python setup.py install')

    def initialize(self):
        self.has_printed = False
        self.lib = None
        self.run_cuda_setup()

    def run_cuda_setup(self):
        self.initialized = True
        self.cuda_setup_log = []

        binary_name = "libbitsandbytes_hip_nocublaslt.so"
        package_dir = Path(__file__).parent
        binary_path = package_dir / binary_name

        try:
            if not binary_path.exists():
                raise Exception('CUDA SETUP: Setup Failed!')
            else:
                self.add_log_entry(f"CUDA SETUP: Loading binary {binary_path}...")
                self.lib = ct.cdll.LoadLibrary(binary_path)
        except Exception as ex:
            self.add_log_entry(str(ex))
            self.print_log_stack()

    def add_log_entry(self, msg, is_warning=False):
        self.cuda_setup_log.append((msg, is_warning))

    def print_log_stack(self):
        for msg, is_warning in self.cuda_setup_log:
            if is_warning:
                warn(msg)
            else:
                print(msg)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance


setup = CUDASetup.get_instance()
if setup.initialized != True:
    setup.run_cuda_setup()

lib = setup.lib
try:
    if lib is None and torch.cuda.is_available():
        CUDASetup.get_instance().generate_instructions()
        CUDASetup.get_instance().print_log_stack()
        raise RuntimeError('''
        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues''')
    lib.cadam32bit_grad_fp32 # runs on an error if the library could not be found -> COMPILED_WITH_CUDA=False
    lib.get_context.restype = ct.c_void_p
    lib.get_cusparse.restype = ct.c_void_p
    lib.cget_managed_ptr.restype = ct.c_void_p
    COMPILED_WITH_CUDA = True
except AttributeError as ex:
    warn("The installed version of bitsandbytes was compiled without GPU support. "
        "8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.")
    COMPILED_WITH_CUDA = False
    print(str(ex))


# print the setup details after checking for errors so we do not print twice
if 'BITSANDBYTES_NOWELCOME' not in os.environ or str(os.environ['BITSANDBYTES_NOWELCOME']) == '0':
    setup.print_log_stack()
