"""Read only the CUDA devices visible to the server's current allocation."""
import os
import subprocess


def gpu_inventory():
    import torch
    if not torch.cuda.is_available():
        return []
    busy = set()
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                timeout=3, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                uuid, pid = [v.strip() for v in line.split(',')]
                if pid != str(os.getpid()):
                    busy.add(uuid.removeprefix('GPU-'))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    devices = []
    for index in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(index)
            props = torch.cuda.get_device_properties(index)
            uuid = str(getattr(props, 'uuid', '')).removeprefix('GPU-')
            devices.append(dict(device=f'cuda:{index}', name=props.name,
                                idle=free >= total * .9 and uuid not in busy))
        except (RuntimeError, AssertionError):
            # An unreadable device is never considered free.
            continue
    return devices
