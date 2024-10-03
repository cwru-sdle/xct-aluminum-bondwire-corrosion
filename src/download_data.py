import os
import tarfile
from osfclient import OSF
from config import DataConfig

def download_files(project_id: str, output_dir: str) -> None:
    """
    Download .tar.gz files from an OSF project and unpacks them.
    
    Args:
        project_id (str): The OSF project ID.
        output_dir (str): Directory to save downloaded and unpacked files.
    """
    osf = OSF()
    project = osf.project(project_id)
    
    files_to_download = []
    for storage in project.storages:
        for file in storage.files:
            if file.path.endswith('.tar.gz'):
                filename = os.path.basename(file.path)
                try:
                    # download .tar.gz
                    output_path = os.path.join(output_dir, filename)
                    with open(output_path, 'wb') as f:
                        file.write_to(f)
                    
                    # unpack the .tar.gz file
                    with tarfile.open(output_path, 'r:gz') as tar:
                        tar.extractall(output_dir)
                    
                    # delete the original .tar.gz file
                    os.remove(output_path)
                except Exception as e:
                    print(f'Error downloading or unpacking {filename}: {str(e)}')
    print('Download and unpacking complete!')

def main():
    config = DataConfig()    
    download_files(config.project_id, config.download_dir)

if __name__ == '__main__':
    main()