import os
import argparse
from osfclient import OSF

def download_osf_files(project_id, output_dir):
    # Initialize the OSF client
    osf = OSF()
    
    # Fetch the project
    project = osf.project(project_id)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through the files in the project
    for storage in project.storages:
        for file in storage.files:
            # Create subdirectories if necessary
            file_path = os.path.join(output_dir, file.path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Download the file
            print(f"Downloading: {file.path}")
            with open(file_path, 'wb') as f:
                file.write_to(f)

def main():
    parser = argparse.ArgumentParser(description='Download files from an OSF project.')
    parser.add_argument('project_id', help='The OSF project ID')
    parser.add_argument('--output', default='/data/raw', help='Output directory for downloaded files')
    
    args = parser.parse_args()
    
    download_osf_files(args.project_id, args.output)
    print("Download complete!")

if __name__ == '__main__':
    main()