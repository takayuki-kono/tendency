import sys
import os

# Add temp_imagedl to path
sys.path.append(os.path.join(os.getcwd(), 'temp_imagedl'))

try:
    from imagedl.modules.sources import BingImageClient, GoogleImageClient
    print("Imports successful.")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_bing():
    print("\n--- Testing BingImageClient ---")
    client = BingImageClient()
    # Search for a few images
    try:
        image_infos = client.search('cat', search_limits=5, num_threadings=1)
        print(f"Found {len(image_infos)} images.")
        for info in image_infos:
            print(f"URL: {info['candidate_urls'][0]}")
    except Exception as e:
        print(f"Bing search failed: {e}")

def test_google():
    print("\n--- Testing GoogleImageClient ---")
    client = GoogleImageClient()
    try:
        image_infos = client.search('cat', search_limits=5, num_threadings=1)
        print(f"Found {len(image_infos)} images.")
        for info in image_infos:
            print(f"URL: {info['candidate_urls'][0]}")
    except Exception as e:
        print(f"Google search failed: {e}")

if __name__ == "__main__":
    test_bing()
    test_google()
