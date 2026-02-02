
import sys
import logging
from imagedl import imagedl

# Minimal setup to inspect search results
try:
    client = imagedl.ImageClient(
        image_source='GoogleImageClient',
        init_image_client_cfg={'work_dir': 'temp_debug_chk', 'max_retries': 1},
        search_limits=1,
        num_threadings=1
    )
    # Perform a very small search
    print("Searching...")
    image_infos = client.search('test', filters={'size': 'large'})
    
    if image_infos and len(image_infos) > 0:
        print("First item keys:", image_infos[0].keys())
        print("First item content:", image_infos[0])
    else:
        print("No results found or format unexpected.")

except Exception as e:
    print(f"Error: {e}")
