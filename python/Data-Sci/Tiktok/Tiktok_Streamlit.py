## Importing TikTok Python SDK and JSON to export data
from TikTokApi import TikTokApi as tiktok
import json

## Cookie Data
verifyFp = ""
## Set up instance
api = tiktok.get_instance(custom_verifyFp=verifyFp, use_test_endpoint=True)
