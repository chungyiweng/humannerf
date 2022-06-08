import os

def list_files(folder_path, exts=None, keyword=None):
    file_list = [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, fname))
                    and (exts is None or any(fname.endswith(ext) for ext in exts))
                    and (keyword is None or (fname.find(keyword)!=-1))
        ]
    file_list = sorted(file_list)

    return file_list

def split_path(file_path):
    file_dir, file_name = os.path.split(file_path)
    file_base_name, file_ext = os.path.splitext(file_name)
    return file_dir, file_base_name, file_ext
    