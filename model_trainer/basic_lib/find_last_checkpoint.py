from pathlib import Path
import re


def get_latest_checkpoint_path(work_dir):
    if not isinstance(work_dir, Path):
        work_dir = Path(work_dir)
    if not work_dir.exists():
        return None

    last_step = -1
    last_ckpt_name = None

    for ckpt in work_dir.glob('model_ckpt_steps_*.ckpt'):
        search = re.search(r'steps_\d+', ckpt.name)
        if search:
            step = int(search.group(0)[6:])
            if step > last_step:
                last_step = step
                last_ckpt_name = str(ckpt)

    return last_ckpt_name if last_ckpt_name is not None else None
