from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File
import os

# Replace with your SharePoint site URL and credentials
site_url = "https://uniwa-my.sharepoint.com/my?id=%2Fpersonal%2F24341496%5Fstudent%5Fuwa%5Fedu%5Fau%2FDocuments%2FDesktop%2FDataset%2FBrats2023%2FASNR%2DMICCAI%2DBraTS2023%2DGLI%2DChallenge%2DTrainingData%2FASNR%2DMICCAI%2DBraTS2023%2DGLI%2DChallenge%2DTrainingData"
username = "danish.ali@research.uwa.edu.au"
password = "Sharepoint@2800528"
file_url = "https://uniwa-my.sharepoint.com/:f:/r/personal/24341496_student_uwa_edu_au/Documents/Desktop/Dataset/Brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData?csf=1&web=1&e=RLe1gE"
local_path = "https://uniwa-my.sharepoint.com/:f:/r/personal/24341496_student_uwa_edu_au/Documents/Desktop/Dataset/Brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000?csf=1&web=1&e=ZqbA2J"

# Authenticate
ctx_auth = AuthenticationContext(site_url)
if ctx_auth.acquire_token_for_user(username, password):
    ctx = ClientContext(site_url, ctx_auth)
    response = File.open_binary(ctx, file_url)
    
    # Save the file locally
    with open(local_path, "wb") as local_file:
        local_file.write(response.content)
    print(f"File downloaded and saved to {local_path}")
else:
    print("Authentication failed")
