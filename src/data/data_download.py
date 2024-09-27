import os
import argparse
from tqdm import tqdm
from osfclient import OSF

def download_osf_files(project_id: str, output_dir: str, file_extension: str = '.jpg') -> None:
    """
    Download files from an OSF project.
    
    Args:
    project_id (str): The OSF project ID.
    output_dir (str): Directory to save downloaded files.
    file_extension (str): File extension to filter (default is '.jpg').
    """
    osf = OSF()
    project = osf.project(project_id)
    
    # ensure output directories exist
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    
    files_to_download = []
    for storage in project.storages:
        for file in storage.files:
            # only collect images in image or mask directory with extension
            if file.path.endswith(file_extension):
                if 'Images' in file.path:
                    subdir = 'images'
                elif 'Masks' in file.path:
                    subdir = 'masks'
                else:
                    subdir = None
                
                if subdir:
                    filename = os.path.basename(file.path)
                    output_path = os.path.join(output_dir, subdir, filename)
                    files_to_download.append((file, output_path))

    # download files with progress bar
    for file, output_path in tqdm(files_to_download, desc='Downloading files'):
        try:
            with open(output_path, 'wb') as f:
                file.write_to(f)
        except Exception as e:
            print(f'Error downloading {file.path}: {str(e)}')

    print(f'Downloaded {len(files_to_download)} files.')

def main():
    parser = argparse.ArgumentParser(description='Download files from OSF project.')
    parser.add_argument('--project_id', type=str, default='uzxpc', help='OSF project ID')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory for downloaded files')
    parser.add_argument('--file_extension', type=str, default='.jpg', help='File extension to download (default: .jpg)')
    
    args = parser.parse_args()
    
    download_osf_files(args.project_id, args.output_dir, args.file_extension)
    print('Download complete!')

if __name__ == '__main__':
    main()