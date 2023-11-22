import datetime
import pathlib
import shutil


def code_saver(save_list: list, work_dir):
    work_dir = pathlib.Path(str(work_dir))
    code_dir = work_dir / 'codes' / datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    code_dir.mkdir(exist_ok=True, parents=True)
    for c in save_list:
        c = pathlib.Path(c)
        if c.is_file():
            shutil.copy(c, code_dir / c, )

        if c.is_dir():
            shutil.copytree(c, code_dir / c, dirs_exist_ok=True)
    print(f'| Copied codes to {code_dir}.')
